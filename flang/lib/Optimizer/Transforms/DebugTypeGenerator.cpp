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

static mlir::LLVM::DITypeAttr genPlaceholderType(mlir::MLIRContext *context) {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, "void", 32, 1);
}

static mlir::LLVM::DITypeAttr genBasicType(mlir::MLIRContext *context,
                                           mlir::StringAttr name,
                                           unsigned bitSize,
                                           unsigned decoding) {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, name, bitSize, decoding);
}

mlir::LLVM::DITypeAttr
DebugTypeGenerator::convertType(mlir::Type Ty, mlir::LLVM::DIFileAttr fileAttr,
                                mlir::LLVM::DIScopeAttr scope,
                                mlir::Location loc) {
  mlir::MLIRContext *context = module.getContext();
  if (Ty.isIntOrIndex()) {
    return genBasicType(context, mlir::StringAttr::get(context, "integer"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_signed);
  } else if (Ty.isa<mlir::FloatType>() || Ty.isa<fir::RealType>()) {
    return genBasicType(context, mlir::StringAttr::get(context, "real"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_float);
  } else if (auto logTy = Ty.dyn_cast_or_null<fir::LogicalType>()) {
    return genBasicType(context,
                        mlir::StringAttr::get(context, logTy.getMnemonic()),
                        kindMapping.getLogicalBitsize(logTy.getFKind()),
                        llvm::dwarf::DW_ATE_boolean);
  } else {
    // FIXME: These types are currently unhandled. We are generating a
    // placeholder type to allow us to test supported bits.
    return genPlaceholderType(context);
  }
}

} // namespace fir
