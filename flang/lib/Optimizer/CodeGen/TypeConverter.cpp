//===-- TypeConverter.cpp -- type conversion --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "flang-type-conversion"

#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "DescriptorModel.h"
#include "flang/Optimizer/Builder/Todo.h" // remove when TODO's are done
#include "flang/Optimizer/CodeGen/TBAABuilder.h"
#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

namespace fir {

LLVMTypeConverter::LLVMTypeConverter(mlir::ModuleOp module, bool applyTBAA,
                                     bool forceUnifiedTBAATree)
    : mlir::LLVMTypeConverter(module.getContext(),
                              [&] {
                                mlir::LowerToLLVMOptions options(
                                    module.getContext());
                                options.useOpaquePointers = false;
                                return options;
                              }()),
      kindMapping(getKindMapping(module)),
      specifics(CodeGenSpecifics::get(module.getContext(),
                                      getTargetTriple(module),
                                      getKindMapping(module))),
      tbaaBuilder(std::make_unique<TBAABuilder>(module->getContext(), applyTBAA,
                                                forceUnifiedTBAATree)) {
  LLVM_DEBUG(llvm::dbgs() << "FIR type converter\n");

  // Each conversion should return a value of type mlir::Type.
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
  addConversion(
      [&](fir::ComplexType cmplx) { return convertComplexType(cmplx); });
  addConversion([&](fir::FieldType field) {
    // Convert to i32 because of LLVM GEP indexing restriction.
    return mlir::IntegerType::get(field.getContext(), 32);
  });
  addConversion([&](HeapType heap) { return convertPointerLike(heap); });
  addConversion([&](fir::IntegerType intTy) {
    return mlir::IntegerType::get(
        &getContext(), kindMapping.getIntegerBitsize(intTy.getFKind()));
  });
  addConversion([&](fir::LenType field) {
    // Get size of len paramter from the descriptor.
    return getModel<Fortran::runtime::typeInfo::TypeParameterValue>()(
        &getContext());
  });
  addConversion([&](fir::LogicalType boolTy) {
    return mlir::IntegerType::get(
        &getContext(), kindMapping.getLogicalBitsize(boolTy.getFKind()));
  });
  addConversion([&](fir::LLVMPointerType pointer) {
    return convertPointerLike(pointer);
  });
  addConversion(
      [&](fir::PointerType pointer) { return convertPointerLike(pointer); });
  addConversion(
      [&](fir::RecordType derived, llvm::SmallVectorImpl<mlir::Type> &results) {
        return convertRecordType(derived, results);
      });
  addConversion(
      [&](fir::RealType real) { return convertRealType(real.getFKind()); });
  addConversion(
      [&](fir::ReferenceType ref) { return convertPointerLike(ref); });
  addConversion([&](fir::SequenceType sequence) {
    return convertSequenceType(sequence);
  });
  addConversion([&](fir::TypeDescType tdesc) {
    return convertTypeDescType(tdesc.getContext());
  });
  addConversion([&](fir::VectorType vecTy) {
    return mlir::VectorType::get(llvm::ArrayRef<int64_t>(vecTy.getLen()),
                                 convertType(vecTy.getEleTy()));
  });
  addConversion([&](mlir::TupleType tuple) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << tuple << '\n');
    llvm::SmallVector<mlir::Type> members;
    for (auto mem : tuple.getTypes()) {
      // Prevent fir.box from degenerating to a pointer to a descriptor in the
      // context of a tuple type.
      if (auto box = mem.dyn_cast<fir::BaseBoxType>())
        members.push_back(convertBoxTypeAsStruct(box));
      else
        members.push_back(convertType(mem).cast<mlir::Type>());
    }
    return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members,
                                                  /*isPacked=*/false);
  });
  addConversion([&](mlir::NoneType none) {
    return mlir::LLVM::LLVMStructType::getLiteral(
        none.getContext(), std::nullopt, /*isPacked=*/false);
  });
  // FIXME: https://reviews.llvm.org/D82831 introduced an automatic
  // materialization of conversion around function calls that is not working
  // well with fir lowering to llvm (incorrect llvm.mlir.cast are inserted).
  // Workaround until better analysis: register a handler that does not insert
  // any conversions.
  addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
  // Similar FIXME workaround here (needed for compare.fir/select-type.fir
  // as well as rebox-global.fir tests). This is needed to cope with the
  // the fact that codegen does not lower some operation results to the LLVM
  // type produced by this LLVMTypeConverter. For instance, inside FIR
  // globals, fir.box are lowered to llvm.struct, while the fir.box type
  // conversion translates it into an llvm.ptr<llvm.struct<>> because
  // descriptors are manipulated in memory outside of global initializers
  // where this is not possible. Hence, MLIR inserts
  // builtin.unrealized_conversion_cast after the translation of operations
  // producing fir.box in fir.global codegen. addSourceMaterialization and
  // addTargetMaterialization allow ignoring these ops and removing them
  // after codegen assuming the type discrepencies are intended (like for
  // fir.box inside globals).
  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
}

// i32 is used here because LLVM wants i32 constants when indexing into struct
// types. Indexing into other aggregate types is more flexible.
mlir::Type LLVMTypeConverter::offsetType() const {
  return mlir::IntegerType::get(&getContext(), 32);
}

// i64 can be used to index into aggregates like arrays
mlir::Type LLVMTypeConverter::indexType() const {
  return mlir::IntegerType::get(&getContext(), 64);
}

// fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
std::optional<mlir::LogicalResult> LLVMTypeConverter::convertRecordType(
    fir::RecordType derived, llvm::SmallVectorImpl<mlir::Type> &results) {
  auto name = derived.getName();
  auto st = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);

  auto &callStack = getCurrentThreadRecursiveStack();
  if (llvm::count(callStack, derived)) {
    results.push_back(st);
    return mlir::success();
  }
  callStack.push_back(derived);
  auto popConversionCallStack =
      llvm::make_scope_exit([&callStack]() { callStack.pop_back(); });

  llvm::SmallVector<mlir::Type> members;
  for (auto mem : derived.getTypeList()) {
    // Prevent fir.box from degenerating to a pointer to a descriptor in the
    // context of a record type.
    if (auto box = mem.second.dyn_cast<fir::BaseBoxType>())
      members.push_back(convertBoxTypeAsStruct(box));
    else
      members.push_back(convertType(mem.second).cast<mlir::Type>());
  }
  if (mlir::failed(st.setBody(members, /*isPacked=*/false)))
    return mlir::failure();
  results.push_back(st);
  return mlir::success();
}

// Is an extended descriptor needed given the element type of a fir.box type ?
// Extended descriptors are required for derived types.
bool LLVMTypeConverter::requiresExtendedDesc(mlir::Type boxElementType) const {
  auto eleTy = fir::unwrapSequenceType(boxElementType);
  return eleTy.isa<fir::RecordType>();
}

// This corresponds to the descriptor as defined in ISO_Fortran_binding.h and
// the addendum defined in descriptor.h.
mlir::Type LLVMTypeConverter::convertBoxType(BaseBoxType box, int rank) const {
  // (base_addr*, elem_len, version, rank, type, attribute, f18Addendum, [dim]
  llvm::SmallVector<mlir::Type> dataDescFields;
  mlir::Type ele = box.getEleTy();
  // remove fir.heap/fir.ref/fir.ptr
  if (auto removeIndirection = fir::dyn_cast_ptrEleTy(ele))
    ele = removeIndirection;
  auto eleTy = convertType(ele);
  // base_addr*
  if (ele.isa<SequenceType>() && eleTy.isa<mlir::LLVM::LLVMPointerType>())
    dataDescFields.push_back(eleTy);
  else
    dataDescFields.push_back(mlir::LLVM::LLVMPointerType::get(eleTy));
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
  // f18Addendum
  dataDescFields.push_back(
      getDescFieldTypeModel<kF18AddendumPosInBox>()(&getContext()));
  // [dims]
  if (rank == unknownRank()) {
    if (auto seqTy = ele.dyn_cast<SequenceType>())
      rank = seqTy.getDimension();
    else
      rank = 0;
  }
  if (rank > 0) {
    auto rowTy = getDescFieldTypeModel<kDimsPosInBox>()(&getContext());
    dataDescFields.push_back(mlir::LLVM::LLVMArrayType::get(rowTy, rank));
  }
  // opt-type-ptr: i8* (see fir.tdesc)
  if (requiresExtendedDesc(ele) || fir::isUnlimitedPolymorphicType(box)) {
    dataDescFields.push_back(
        getExtendedDescFieldTypeModel<kOptTypePtrPosInBox>()(&getContext()));
    auto rowTy =
        getExtendedDescFieldTypeModel<kOptRowTypePosInBox>()(&getContext());
    dataDescFields.push_back(mlir::LLVM::LLVMArrayType::get(rowTy, 1));
    if (auto recTy = fir::unwrapSequenceType(ele).dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() > 0) {
        // The descriptor design needs to be clarified regarding the number of
        // length parameters in the addendum. Since it can change for
        // polymorphic allocatables, it seems all length parameters cannot
        // always possibly be placed in the addendum.
        TODO_NOLOC("extended descriptor derived with length parameters");
        unsigned numLenParams = recTy.getNumLenParams();
        dataDescFields.push_back(
            mlir::LLVM::LLVMArrayType::get(rowTy, numLenParams));
      }
  }
  // TODO: send the box type and the converted LLVM structure layout
  // to tbaaBuilder for proper creation of TBAATypeDescriptorOp.
  return mlir::LLVM::LLVMPointerType::get(
      mlir::LLVM::LLVMStructType::getLiteral(&getContext(), dataDescFields,
                                             /*isPacked=*/false));
}

/// Convert fir.box type to the corresponding llvm struct type instead of a
/// pointer to this struct type.
mlir::Type LLVMTypeConverter::convertBoxTypeAsStruct(BaseBoxType box) const {
  return convertBoxType(box)
      .cast<mlir::LLVM::LLVMPointerType>()
      .getElementType();
}

// fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
mlir::Type LLVMTypeConverter::convertBoxProcType(BoxProcType boxproc) const {
  auto funcTy = convertType(boxproc.getEleTy());
  auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(&getContext(), 8));
  llvm::SmallVector<mlir::Type, 2> tuple = {funcTy, i8PtrTy};
  return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), tuple,
                                                /*isPacked=*/false);
}

unsigned LLVMTypeConverter::characterBitsize(fir::CharacterType charTy) const {
  return kindMapping.getCharacterBitsize(charTy.getFKind());
}

// fir.char<k,?>  -->  llvm<"ix">          where ix is scaled by kind mapping
// fir.char<k,n>  -->  llvm.array<n x "ix">
mlir::Type LLVMTypeConverter::convertCharType(fir::CharacterType charTy) const {
  auto iTy = mlir::IntegerType::get(&getContext(), characterBitsize(charTy));
  if (charTy.getLen() == fir::CharacterType::unknownLen())
    return iTy;
  return mlir::LLVM::LLVMArrayType::get(iTy, charTy.getLen());
}

// convert a front-end kind value to either a std or LLVM IR dialect type
// fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
mlir::Type LLVMTypeConverter::convertRealType(fir::KindTy kind) const {
  return fir::fromRealTypeID(&getContext(), kindMapping.getRealTypeID(kind),
                             kind);
}

// fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
mlir::Type LLVMTypeConverter::convertSequenceType(SequenceType seq) const {
  auto baseTy = convertType(seq.getEleTy());
  if (characterWithDynamicLen(seq.getEleTy()))
    return mlir::LLVM::LLVMPointerType::get(baseTy);
  auto shape = seq.getShape();
  auto constRows = seq.getConstantRows();
  if (constRows) {
    decltype(constRows) i = constRows;
    for (auto e : shape) {
      baseTy = mlir::LLVM::LLVMArrayType::get(baseTy, e);
      if (--i == 0)
        break;
    }
    if (!seq.hasDynamicExtents())
      return baseTy;
  }
  return mlir::LLVM::LLVMPointerType::get(baseTy);
}

// fir.tdesc<any>  -->  llvm<"i8*">
// TODO: For now use a void*, however pointer identity is not sufficient for
// the f18 object v. class distinction (F2003).
mlir::Type
LLVMTypeConverter::convertTypeDescType(mlir::MLIRContext *ctx) const {
  return mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(&getContext(), 8));
}

// Relay TBAA tag attachment to TBAABuilder.
void LLVMTypeConverter::attachTBAATag(mlir::LLVM::AliasAnalysisOpInterface op,
                                      mlir::Type baseFIRType,
                                      mlir::Type accessFIRType,
                                      mlir::LLVM::GEPOp gep) const {
  tbaaBuilder->attachTBAATag(op, baseFIRType, accessFIRType, gep);
}

} // namespace fir
