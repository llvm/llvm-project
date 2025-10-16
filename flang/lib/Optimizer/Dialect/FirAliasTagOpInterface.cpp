//===-- FirAliasTagOpInterface.cpp ----------------------------------===//
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

#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.cpp.inc"

llvm::LogicalResult
fir::detail::verifyFirAliasTagOpInterface(mlir::Operation *op) {
  auto iface = mlir::cast<FirAliasTagOpInterface>(op);

  mlir::ArrayAttr tags = iface.getTBAATagsOrNull();
  if (!tags)
    return llvm::success();

  for (mlir::Attribute iter : tags)
    if (!mlir::isa<mlir::LLVM::TBAATagAttr>(iter))
      return op->emitOpError("expected op to return array of ")
             << mlir::LLVM::TBAATagAttr::getMnemonic() << " attributes";
  return llvm::success();
}
