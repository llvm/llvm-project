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
#include "flang/Lower/Todo.h" // remove when TODO's are done
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

namespace fir {

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::ModuleOp module)
      : mlir::LLVMTypeConverter(module.getContext()) {
    LLVM_DEBUG(llvm::dbgs() << "FIR type converter\n");

    // Each conversion should return a value of type mlir::Type.
    addConversion(
        [&](fir::RecordType derived) { return convertRecordType(derived); });
    addConversion(
        [&](fir::ReferenceType ref) { return convertPointerLike(ref); });
    addConversion(
        [&](SequenceType sequence) { return convertSequenceType(sequence); });
    addConversion([&](mlir::TupleType tuple) {
      LLVM_DEBUG(llvm::dbgs() << "type convert: " << tuple << '\n');
      llvm::SmallVector<mlir::Type> inMembers;
      tuple.getFlattenedTypes(inMembers);
      llvm::SmallVector<mlir::Type> members;
      for (auto mem : inMembers)
        members.push_back(convertType(mem).cast<mlir::Type>());
      return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members,
                                                    /*isPacked=*/false);
    });
  }

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  mlir::Type convertRecordType(fir::RecordType derived) {
    auto name = derived.getName();
    auto st = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
    llvm::SmallVector<mlir::Type> members;
    for (auto mem : derived.getTypeList()) {
      members.push_back(convertType(mem.second).cast<mlir::Type>());
    }
    if (mlir::succeeded(st.setBody(members, /*isPacked=*/false)))
      return st;
    return mlir::Type();
  }

  template <typename A>
  mlir::Type convertPointerLike(A &ty) {
    mlir::Type eleTy = ty.getEleTy();
    // A sequence type is a special case. A sequence of runtime size on its
    // interior dimensions lowers to a memory reference. In that case, we
    // degenerate the array and do not want a the type to become `T**` but
    // merely `T*`.
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (!seqTy.hasConstantShape() ||
          characterWithDynamicLen(seqTy.getEleTy())) {
        if (seqTy.hasConstantInterior())
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
      if (seq.hasConstantShape())
        return baseTy;
    }
    return mlir::LLVM::LLVMPointerType::get(baseTy);
  }
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
