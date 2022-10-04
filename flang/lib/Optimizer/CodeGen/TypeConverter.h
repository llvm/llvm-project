//===-- TypeConverter.h -- type conversion ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
#define FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H

#include "DescriptorModel.h"
#include "Target.h"
#include "flang/Optimizer/Builder/Todo.h" // remove when TODO's are done
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "llvm/Support/Debug.h"

// Position of the different values in a `fir.box`.
static constexpr unsigned kAddrPosInBox = 0;
static constexpr unsigned kElemLenPosInBox = 1;
static constexpr unsigned kVersionPosInBox = 2;
static constexpr unsigned kRankPosInBox = 3;
static constexpr unsigned kTypePosInBox = 4;
static constexpr unsigned kAttributePosInBox = 5;
static constexpr unsigned kF18AddendumPosInBox = 6;
static constexpr unsigned kDimsPosInBox = 7;
static constexpr unsigned kOptTypePtrPosInBox = 8;
static constexpr unsigned kOptRowTypePosInBox = 9;

// Position of the different values in [dims]
static constexpr unsigned kDimLowerBoundPos = 0;
static constexpr unsigned kDimExtentPos = 1;
static constexpr unsigned kDimStridePos = 2;

namespace fir {

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::ModuleOp module)
      : mlir::LLVMTypeConverter(module.getContext()),
        kindMapping(getKindMapping(module)),
        specifics(CodeGenSpecifics::get(module.getContext(),
                                        getTargetTriple(module),
                                        getKindMapping(module))) {
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
      return llvm::None;
    });
    addConversion([&](fir::ClassType classTy) {
      TODO_NOLOC("fir.class type conversion");
      return llvm::None;
    });
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
    addConversion([&](fir::RecordType derived,
                      llvm::SmallVectorImpl<mlir::Type> &results,
                      llvm::ArrayRef<mlir::Type> callStack) {
      return convertRecordType(derived, results, callStack);
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
        if (auto box = mem.dyn_cast<fir::BoxType>())
          members.push_back(convertBoxTypeAsStruct(box));
        else
          members.push_back(convertType(mem).cast<mlir::Type>());
      }
      return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members,
                                                    /*isPacked=*/false);
    });
    addConversion([&](mlir::NoneType none) {
      return mlir::LLVM::LLVMStructType::getLiteral(
          none.getContext(), llvm::None, /*isPacked=*/false);
    });
    // FIXME: https://reviews.llvm.org/D82831 introduced an automatic
    // materialization of conversion around function calls that is not working
    // well with fir lowering to llvm (incorrect llvm.mlir.cast are inserted).
    // Workaround until better analysis: register a handler that does not insert
    // any conversions.
    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> llvm::Optional<mlir::Value> {
          if (inputs.size() != 1)
            return llvm::None;
          return inputs[0];
        });
    // Similar FIXME workaround here (needed for compare.fir/select-type.fir
    // tests).
    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> llvm::Optional<mlir::Value> {
          if (inputs.size() != 1)
            return llvm::None;
          return inputs[0];
        });
  }

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible.
  mlir::Type offsetType() { return mlir::IntegerType::get(&getContext(), 32); }

  // i64 can be used to index into aggregates like arrays
  mlir::Type indexType() { return mlir::IntegerType::get(&getContext(), 64); }

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  llvm::Optional<mlir::LogicalResult>
  convertRecordType(fir::RecordType derived,
                    llvm::SmallVectorImpl<mlir::Type> &results,
                    llvm::ArrayRef<mlir::Type> callStack) {
    auto name = derived.getName();
    auto st = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
    if (llvm::count(callStack, derived) > 1) {
      results.push_back(st);
      return mlir::success();
    }
    llvm::SmallVector<mlir::Type> members;
    for (auto mem : derived.getTypeList()) {
      // Prevent fir.box from degenerating to a pointer to a descriptor in the
      // context of a record type.
      if (auto box = mem.second.dyn_cast<fir::BoxType>())
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
  bool requiresExtendedDesc(mlir::Type boxElementType) {
    auto eleTy = fir::unwrapSequenceType(boxElementType);
    return eleTy.isa<fir::RecordType>();
  }

  // Magic value to indicate we do not know the rank of an entity, either
  // because it is assumed rank or because we have not determined it yet.
  static constexpr int unknownRank() { return -1; }

  // This corresponds to the descriptor as defined in ISO_Fortran_binding.h and
  // the addendum defined in descriptor.h.
  mlir::Type convertBoxType(BoxType box, int rank = unknownRank()) {
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
    if (requiresExtendedDesc(ele)) {
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
    return mlir::LLVM::LLVMPointerType::get(
        mlir::LLVM::LLVMStructType::getLiteral(&getContext(), dataDescFields,
                                               /*isPacked=*/false));
  }

  /// Convert fir.box type to the corresponding llvm struct type instead of a
  /// pointer to this struct type.
  mlir::Type convertBoxTypeAsStruct(BoxType box) {
    return convertBoxType(box)
        .cast<mlir::LLVM::LLVMPointerType>()
        .getElementType();
  }

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  mlir::Type convertBoxProcType(BoxProcType boxproc) {
    auto funcTy = convertType(boxproc.getEleTy());
    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));
    llvm::SmallVector<mlir::Type, 2> tuple = {funcTy, i8PtrTy};
    return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), tuple,
                                                  /*isPacked=*/false);
  }

  unsigned characterBitsize(fir::CharacterType charTy) {
    return kindMapping.getCharacterBitsize(charTy.getFKind());
  }

  // fir.char<k,?>  -->  llvm<"ix">          where ix is scaled by kind mapping
  // fir.char<k,n>  -->  llvm.array<n x "ix">
  mlir::Type convertCharType(fir::CharacterType charTy) {
    auto iTy = mlir::IntegerType::get(&getContext(), characterBitsize(charTy));
    if (charTy.getLen() == fir::CharacterType::unknownLen())
      return iTy;
    return mlir::LLVM::LLVMArrayType::get(iTy, charTy.getLen());
  }

  // Use the target specifics to figure out how to map complex to LLVM IR. The
  // use of complex values in function signatures is handled before conversion
  // to LLVM IR dialect here.
  //
  // fir.complex<T> | std.complex<T>    --> llvm<"{t,t}">
  template <typename C>
  mlir::Type convertComplexType(C cmplx) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << cmplx << '\n');
    auto eleTy = cmplx.getElementType();
    return convertType(specifics->complexMemoryType(eleTy));
  }

  template <typename A>
  mlir::Type convertPointerLike(A &ty) {
    mlir::Type eleTy = ty.getEleTy();
    // A sequence type is a special case. A sequence of runtime size on its
    // interior dimensions lowers to a memory reference. In that case, we
    // degenerate the array and do not want a the type to become `T**` but
    // merely `T*`.
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (seqTy.hasDynamicExtents() ||
          characterWithDynamicLen(seqTy.getEleTy())) {
        if (seqTy.getConstantRows() > 0)
          return convertType(seqTy);
        eleTy = seqTy.getEleTy();
      }
    }
    // fir.ref<fir.box> is a special case because fir.box type is already
    // a pointer to a Fortran descriptor at the LLVM IR level. This implies
    // that a fir.ref<fir.box>, that is the address of fir.box is actually
    // the same as a fir.box at the LLVM level.
    // The distinction is kept in fir to denote when a descriptor is expected
    // to be mutable (fir.ref<fir.box>) and when it is not (fir.box).
    if (eleTy.isa<fir::BoxType>())
      return convertType(eleTy);

    return mlir::LLVM::LLVMPointerType::get(convertType(eleTy));
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  mlir::Type convertRealType(fir::KindTy kind) {
    return fir::fromRealTypeID(&getContext(), kindMapping.getRealTypeID(kind),
                               kind);
  }

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  mlir::Type convertSequenceType(SequenceType seq) {
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
  mlir::Type convertTypeDescType(mlir::MLIRContext *ctx) {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));
  }

  KindMapping &getKindMap() { return kindMapping; }

private:
  KindMapping kindMapping;
  std::unique_ptr<CodeGenSpecifics> specifics;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
