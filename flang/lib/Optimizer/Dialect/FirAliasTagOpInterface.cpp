//===-- FirAliasTagOpInterface.cpp ----------------------------------===//
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

#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"

#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.cpp.inc"

llvm::LogicalResult
fir::detail::verifyFirAliasTagOpInterface(aiir::Operation *op) {
  auto iface = aiir::cast<FirAliasTagOpInterface>(op);

  aiir::ArrayAttr tags = iface.getTBAATagsOrNull();
  if (!tags)
    return llvm::success();

  for (aiir::Attribute iter : tags)
    if (!aiir::isa<aiir::LLVM::TBAATagAttr>(iter))
      return op->emitOpError("expected op to return array of ")
             << aiir::LLVM::TBAATagAttr::getMnemonic() << " attributes";
  return llvm::success();
}
