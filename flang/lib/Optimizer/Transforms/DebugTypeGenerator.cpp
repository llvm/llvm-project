//===-- DebugTypeGenerator.cpp -- type conversion ---------------*- C++ -*-===//
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

#define DEBUG_TYPE "flang-debug-type-generator"

#include "DebugTypeGenerator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"

namespace fir {

DebugTypeGenerator::DebugTypeGenerator(mlir::ModuleOp m)
    : module(m), kindMapping(getKindMapping(m)) {
  LLVM_DEBUG(llvm::dbgs() << "DITypeAttr generator\n");
}

static mlir::LLVM::DITypeAttr genBasicType(mlir::MLIRContext *context,
                                           mlir::StringAttr name,
                                           unsigned bitSize,
                                           unsigned decoding) {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, name, bitSize, decoding);
}

static mlir::LLVM::DITypeAttr genPlaceholderType(mlir::MLIRContext *context) {
  return genBasicType(context, mlir::StringAttr::get(context, "integer"), 32,
                      llvm::dwarf::DW_ATE_signed);
}

mlir::LLVM::DITypeAttr DebugTypeGenerator::convertSequenceType(
    fir::SequenceType seqTy, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, mlir::Location loc) {

  mlir::MLIRContext *context = module.getContext();
  // FIXME: Only fixed sizes arrays handled at the moment.
  if (seqTy.hasDynamicExtents())
    return genPlaceholderType(context);

  llvm::SmallVector<mlir::LLVM::DINodeAttr> elements;
  mlir::LLVM::DITypeAttr elemTy =
      convertType(seqTy.getEleTy(), fileAttr, scope, loc);

  for (fir::SequenceType::Extent dim : seqTy.getShape()) {
    auto intTy = mlir::IntegerType::get(context, 64);
    // FIXME: Only supporting lower bound of 1 at the moment. The
    // 'SequenceType' has information about the shape but not the shift. In
    // cases where the conversion originated during the processing of
    // 'DeclareOp', it may be possible to pass on this information. But the
    // type conversion should ideally be based on what information present in
    // the type class so that it works from everywhere (e.g. when it is part
    // of a module or a derived type.)
    auto countAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, dim));
    auto lowerAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, 1));
    auto subrangeTy = mlir::LLVM::DISubrangeAttr::get(
        context, countAttr, lowerAttr, nullptr, nullptr);
    elements.push_back(subrangeTy);
  }
  // Apart from arrays, the `DICompositeTypeAttr` is used for other things like
  // structure types. Many of its fields which are not applicable to arrays
  // have been set to some valid default values.

  return mlir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type, /*recursive id*/ {},
      /* name */ nullptr, /* file */ nullptr, /* line */ 0, /* scope */ nullptr,
      elemTy, mlir::LLVM::DIFlags::Zero, /* sizeInBits */ 0,
      /*alignInBits*/ 0, elements, /* dataLocation */ nullptr,
      /* rank */ nullptr, /* allocated */ nullptr,
      /* associated */ nullptr);
}

mlir::LLVM::DITypeAttr
DebugTypeGenerator::convertType(mlir::Type Ty, mlir::LLVM::DIFileAttr fileAttr,
                                mlir::LLVM::DIScopeAttr scope,
                                mlir::Location loc) {
  mlir::MLIRContext *context = module.getContext();
  if (Ty.isInteger()) {
    return genBasicType(context, mlir::StringAttr::get(context, "integer"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_signed);
  } else if (mlir::isa<mlir::FloatType>(Ty)) {
    return genBasicType(context, mlir::StringAttr::get(context, "real"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_float);
  } else if (auto realTy = mlir::dyn_cast_or_null<fir::RealType>(Ty)) {
    return genBasicType(context, mlir::StringAttr::get(context, "real"),
                        kindMapping.getRealBitsize(realTy.getFKind()),
                        llvm::dwarf::DW_ATE_float);
  } else if (auto logTy = mlir::dyn_cast_or_null<fir::LogicalType>(Ty)) {
    return genBasicType(context,
                        mlir::StringAttr::get(context, logTy.getMnemonic()),
                        kindMapping.getLogicalBitsize(logTy.getFKind()),
                        llvm::dwarf::DW_ATE_boolean);
  } else if (fir::isa_complex(Ty)) {
    unsigned bitWidth;
    if (auto cplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(Ty)) {
      auto floatTy = mlir::cast<mlir::FloatType>(cplxTy.getElementType());
      bitWidth = floatTy.getWidth();
    } else if (auto cplxTy = mlir::dyn_cast_or_null<fir::ComplexType>(Ty)) {
      bitWidth = kindMapping.getRealBitsize(cplxTy.getFKind());
    } else {
      llvm_unreachable("Unhandled complex type");
    }
    return genBasicType(context, mlir::StringAttr::get(context, "complex"),
                        bitWidth * 2, llvm::dwarf::DW_ATE_complex_float);
  } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(Ty)) {
    return convertSequenceType(seqTy, fileAttr, scope, loc);
  } else {
    // FIXME: These types are currently unhandled. We are generating a
    // placeholder type to allow us to test supported bits.
    return genPlaceholderType(context);
  }
}

} // namespace fir
