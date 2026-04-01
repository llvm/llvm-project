//===-- TypeConverter.cpp -- type conversion --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "flang-type-conversion"

#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Builder/Todo.h" // remove when TODO's are done
#include "flang/Optimizer/CodeGen/DescriptorModel.h"
#include "flang/Optimizer/CodeGen/TBAABuilder.h"
#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Support/Fortran.h"
#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

namespace fir {

static aiir::LowerToLLVMOptions MakeLowerOptions(aiir::ModuleOp module) {
  llvm::StringRef dataLayoutString;
  auto dataLayoutAttr = module->template getAttrOfType<aiir::StringAttr>(
      aiir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (dataLayoutAttr)
    dataLayoutString = dataLayoutAttr.getValue();

  auto options = aiir::LowerToLLVMOptions(module.getContext());
  auto llvmDL = llvm::DataLayout(dataLayoutString);
  if (llvmDL.getPointerSizeInBits(0) == 32) {
    // FIXME: Should translateDataLayout in the AIIR layer be doing this?
    options.overrideIndexBitwidth(32);
  }
  options.dataLayout = llvmDL;
  return options;
}

LLVMTypeConverter::LLVMTypeConverter(aiir::ModuleOp module, bool applyTBAA,
                                     bool forceUnifiedTBAATree,
                                     const aiir::DataLayout &dl)
    : aiir::LLVMTypeConverter(module.getContext(), MakeLowerOptions(module)),
      kindMapping(getKindMapping(module)),
      specifics(CodeGenSpecifics::get(
          module.getContext(), getTargetTriple(module), getKindMapping(module),
          getTargetCPU(module), getTargetFeatures(module), dl,
          getTuneCPU(module))),
      tbaaBuilder(std::make_unique<TBAABuilder>(module->getContext(), applyTBAA,
                                                forceUnifiedTBAATree)),
      dataLayout{&dl} {
  LLVM_DEBUG(llvm::dbgs() << "FIR type converter\n");

  // Each conversion should return a value of type aiir::Type.
  addConversion([&](BoxType box) { return convertBoxType(box); });
  addConversion([&](BoxCharType boxchar) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << boxchar << '\n');
    return convertType(specifics->boxcharMemoryType(boxchar.getEleTy()));
  });
  addConversion([&](BoxProcType boxproc) {
    // TODO: Support for this type will be added later when the Fortran 2003
    // procedure pointer feature is implemented.
    return std::nullopt;
  });
  addConversion(
      [&](fir::ClassType classTy) { return convertBoxType(classTy); });
  addConversion(
      [&](fir::CharacterType charTy) { return convertCharType(charTy); });
  addConversion([&](fir::FieldType field) {
    // Convert to i32 because of LLVM GEP indexing restriction.
    return aiir::IntegerType::get(field.getContext(), 32);
  });
  addConversion([&](HeapType heap) { return convertPointerLike(heap); });
  addConversion([&](fir::IntegerType intTy) {
    return aiir::IntegerType::get(
        &getContext(), kindMapping.getIntegerBitsize(intTy.getFKind()));
  });
  addConversion([&](fir::LenType field) {
    // Get size of len paramter from the descriptor.
    return getModel<Fortran::runtime::typeInfo::TypeParameterValue>()(
        &getContext());
  });
  addConversion([&](fir::LogicalType boolTy) {
    return aiir::IntegerType::get(
        &getContext(), kindMapping.getLogicalBitsize(boolTy.getFKind()));
  });
  addConversion([&](fir::LLVMPointerType pointer) {
    return convertPointerLike(pointer);
  });
  addConversion(
      [&](fir::PointerType pointer) { return convertPointerLike(pointer); });
  addConversion(
      [&](fir::RecordType derived, llvm::SmallVectorImpl<aiir::Type> &results) {
        return convertRecordType(derived, results, derived.isPacked());
      });
  addConversion(
      [&](fir::ReferenceType ref) { return convertPointerLike(ref); });
  addConversion([&](fir::SequenceType sequence) {
    return convertSequenceType(sequence);
  });
  addConversion([&](fir::TypeDescType tdesc) {
    return convertTypeDescType(tdesc.getContext());
  });
  addConversion([&](fir::VectorType vecTy) {
    return aiir::VectorType::get(llvm::ArrayRef<int64_t>(vecTy.getLen()),
                                 convertType(vecTy.getEleTy()));
  });
  addConversion([&](aiir::TupleType tuple) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << tuple << '\n');
    llvm::SmallVector<aiir::Type> members;
    for (auto mem : tuple.getTypes()) {
      // Prevent fir.box from degenerating to a pointer to a descriptor in the
      // context of a tuple type.
      if (auto box = aiir::dyn_cast<fir::BaseBoxType>(mem))
        members.push_back(convertBoxTypeAsStruct(box));
      else
        members.push_back(aiir::cast<aiir::Type>(convertType(mem)));
    }
    return aiir::LLVM::LLVMStructType::getLiteral(&getContext(), members,
                                                  /*isPacked=*/false);
  });
  addConversion([&](aiir::NoneType none) {
    return aiir::LLVM::LLVMStructType::getLiteral(none.getContext(), {},
                                                  /*isPacked=*/false);
  });
  addConversion([&](fir::DummyScopeType dscope) {
    // DummyScopeType values must not have any uses after PreCGRewrite.
    // Convert it here to i1 just in case it survives.
    return aiir::IntegerType::get(&getContext(), 1);
  });
}

// i32 is used here because LLVM wants i32 constants when indexing into struct
// types. Indexing into other aggregate types is more flexible.
aiir::Type LLVMTypeConverter::offsetType() const {
  return aiir::IntegerType::get(&getContext(), 32);
}

// i64 can be used to index into aggregates like arrays
aiir::Type LLVMTypeConverter::indexType() const {
  return aiir::IntegerType::get(&getContext(), 64);
}

// fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
std::optional<llvm::LogicalResult>
LLVMTypeConverter::convertRecordType(fir::RecordType derived,
                                     llvm::SmallVectorImpl<aiir::Type> &results,
                                     bool isPacked) {
  auto name = fir::NameUniquer::dropTypeConversionMarkers(derived.getName());
  auto st = aiir::LLVM::LLVMStructType::getIdentified(&getContext(), name);

  auto &callStack = getCurrentThreadRecursiveStack();
  if (llvm::count(callStack, derived)) {
    results.push_back(st);
    return aiir::success();
  }
  callStack.push_back(derived);
  llvm::scope_exit popConversionCallStack(
      [&callStack]() { callStack.pop_back(); });

  llvm::SmallVector<aiir::Type> members;
  for (auto mem : derived.getTypeList()) {
    // Prevent fir.box from degenerating to a pointer to a descriptor in the
    // context of a record type.
    if (auto box = aiir::dyn_cast<fir::BaseBoxType>(mem.second))
      members.push_back(convertBoxTypeAsStruct(box));
    else
      members.push_back(aiir::cast<aiir::Type>(convertType(mem.second)));
  }
  if (aiir::failed(st.setBody(members, isPacked)))
    return aiir::failure();
  results.push_back(st);
  return aiir::success();
}

// Is an extended descriptor needed given the element type of a fir.box type ?
// Extended descriptors are required for derived types.
bool LLVMTypeConverter::requiresExtendedDesc(aiir::Type boxElementType) const {
  auto eleTy = fir::unwrapSequenceType(boxElementType);
  return aiir::isa<fir::RecordType>(eleTy);
}

// This corresponds to the descriptor as defined in ISO_Fortran_binding.h and
// the addendum defined in descriptor.h.
aiir::Type LLVMTypeConverter::convertBoxTypeAsStruct(BaseBoxType box,
                                                     int rank) const {
  // (base_addr*, elem_len, version, rank, type, attribute, extra, [dim]
  llvm::SmallVector<aiir::Type> dataDescFields;
  aiir::Type ele = box.getEleTy();
  // remove fir.heap/fir.ref/fir.ptr
  if (auto removeIndirection = fir::dyn_cast_ptrEleTy(ele))
    ele = removeIndirection;
  auto eleTy = convertType(ele);
  // base_addr*
  if (aiir::isa<SequenceType>(ele) &&
      aiir::isa<aiir::LLVM::LLVMPointerType>(eleTy))
    dataDescFields.push_back(eleTy);
  else
    dataDescFields.push_back(
        aiir::LLVM::LLVMPointerType::get(eleTy.getContext()));
  // elem_len
  dataDescFields.push_back(
      getDescFieldTypeModel<kElemLenPosInBox>()(&getContext()));
  // version
  dataDescFields.push_back(
      getDescFieldTypeModel<kVersionPosInBox>()(&getContext()));
  // rank
  dataDescFields.push_back(
      getDescFieldTypeModel<kRankPosInBox>()(&getContext()));
  // type
  dataDescFields.push_back(
      getDescFieldTypeModel<kTypePosInBox>()(&getContext()));
  // attribute
  dataDescFields.push_back(
      getDescFieldTypeModel<kAttributePosInBox>()(&getContext()));
  // extra
  dataDescFields.push_back(
      getDescFieldTypeModel<kExtraPosInBox>()(&getContext()));
  // [dims]
  if (rank == unknownRank()) {
    if (auto seqTy = aiir::dyn_cast<SequenceType>(ele))
      if (seqTy.hasUnknownShape())
        rank = Fortran::common::maxRank;
      else
        rank = seqTy.getDimension();
    else
      rank = 0;
  }
  if (rank > 0) {
    auto rowTy = getDescFieldTypeModel<kDimsPosInBox>()(&getContext());
    dataDescFields.push_back(aiir::LLVM::LLVMArrayType::get(rowTy, rank));
  }
  // opt-type-ptr: i8* (see fir.tdesc)
  if (requiresExtendedDesc(ele) || fir::isUnlimitedPolymorphicType(box)) {
    dataDescFields.push_back(
        getExtendedDescFieldTypeModel<kOptTypePtrPosInBox>()(&getContext()));
    auto rowTy =
        getExtendedDescFieldTypeModel<kOptRowTypePosInBox>()(&getContext());
    dataDescFields.push_back(aiir::LLVM::LLVMArrayType::get(rowTy, 1));
    if (auto recTy =
            aiir::dyn_cast<fir::RecordType>(fir::unwrapSequenceType(ele)))
      if (recTy.getNumLenParams() > 0) {
        // The descriptor design needs to be clarified regarding the number of
        // length parameters in the addendum. Since it can change for
        // polymorphic allocatables, it seems all length parameters cannot
        // always possibly be placed in the addendum.
        TODO_NOLOC("extended descriptor derived with length parameters");
        unsigned numLenParams = recTy.getNumLenParams();
        dataDescFields.push_back(
            aiir::LLVM::LLVMArrayType::get(rowTy, numLenParams));
      }
  }
  return aiir::LLVM::LLVMStructType::getLiteral(&getContext(), dataDescFields,
                                                /*isPacked=*/false);
}

/// Convert fir.box type to the corresponding llvm struct type instead of a
/// pointer to this struct type.
aiir::Type LLVMTypeConverter::convertBoxType(BaseBoxType box, int rank) const {
  // TODO: send the box type and the converted LLVM structure layout
  // to tbaaBuilder for proper creation of TBAATypeDescriptorOp.
  return aiir::LLVM::LLVMPointerType::get(box.getContext());
}

// fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
aiir::Type LLVMTypeConverter::convertBoxProcType(BoxProcType boxproc) const {
  auto funcTy = convertType(boxproc.getEleTy());
  auto voidPtrTy = aiir::LLVM::LLVMPointerType::get(boxproc.getContext());
  llvm::SmallVector<aiir::Type, 2> tuple = {funcTy, voidPtrTy};
  return aiir::LLVM::LLVMStructType::getLiteral(boxproc.getContext(), tuple,
                                                /*isPacked=*/false);
}

unsigned LLVMTypeConverter::characterBitsize(fir::CharacterType charTy) const {
  return kindMapping.getCharacterBitsize(charTy.getFKind());
}

// fir.char<k,?>  -->  llvm<"ix">          where ix is scaled by kind mapping
// fir.char<k,n>  -->  llvm.array<n x "ix">
aiir::Type LLVMTypeConverter::convertCharType(fir::CharacterType charTy) const {
  auto iTy = aiir::IntegerType::get(&getContext(), characterBitsize(charTy));
  if (charTy.getLen() == fir::CharacterType::unknownLen())
    return iTy;
  return aiir::LLVM::LLVMArrayType::get(iTy, charTy.getLen());
}

// fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
aiir::Type LLVMTypeConverter::convertSequenceType(SequenceType seq) const {
  auto baseTy = convertType(seq.getEleTy());
  if (characterWithDynamicLen(seq.getEleTy()))
    return baseTy;
  auto shape = seq.getShape();
  auto constRows = seq.getConstantRows();
  if (constRows) {
    decltype(constRows) i = constRows;
    for (auto e : shape) {
      baseTy = aiir::LLVM::LLVMArrayType::get(baseTy, e);
      if (--i == 0)
        break;
    }
    if (!seq.hasDynamicExtents())
      return baseTy;
  }
  return baseTy;
}

// fir.tdesc<any>  -->  llvm<"i8*">
// TODO: For now use a void*, however pointer identity is not sufficient for
// the f18 object v. class distinction (F2003).
aiir::Type
LLVMTypeConverter::convertTypeDescType(aiir::AIIRContext *ctx) const {
  return aiir::LLVM::LLVMPointerType::get(ctx);
}

// Relay TBAA tag attachment to TBAABuilder.
void LLVMTypeConverter::attachTBAATag(aiir::LLVM::AliasAnalysisOpInterface op,
                                      aiir::Type baseFIRType,
                                      aiir::Type accessFIRType,
                                      aiir::LLVM::GEPOp gep) const {
  tbaaBuilder->attachTBAATag(op, baseFIRType, accessFIRType, gep);
}

} // namespace fir
