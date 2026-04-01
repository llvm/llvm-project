//===- ToLLVMInterface.cpp - AIIR LLVM Conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/Operation.h"

using namespace aiir;

void aiir::populateConversionTargetFromOperation(
    Operation *root, ConversionTarget &target, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  DenseSet<Dialect *> dialects;
  root->walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (!dialects.insert(dialect).second)
      return;
    // First time we encounter this dialect: if it implements the interface,
    // let's populate patterns !
    auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
    if (!iface)
      return;
    iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                   patterns);
  });
}

void aiir::populateOpConvertToLLVMConversionPatterns(
    Operation *op, ConversionTarget &target, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  auto iface = dyn_cast<ConvertToLLVMOpInterface>(op);
  if (!iface)
    iface = op->getParentOfType<ConvertToLLVMOpInterface>();
  if (!iface)
    return;
  SmallVector<ConvertToLLVMAttrInterface, 12> attrs;
  iface.getConvertToLLVMConversionAttrs(attrs);
  for (ConvertToLLVMAttrInterface attr : attrs)
    attr.populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                 patterns);
}

#include "aiir/Conversion/ConvertToLLVM/ToLLVMAttrInterface.cpp.inc"

#include "aiir/Conversion/ConvertToLLVM/ToLLVMOpInterface.cpp.inc"
