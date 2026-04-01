//===- FuncToEmitCPass.cpp - Func to EmitC Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Func dialect to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/FuncToEmitC/FuncToEmitCPass.h"

#include "aiir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTFUNCTOEMITC
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct ConvertFuncToEmitC
    : public impl::ConvertFuncToEmitCBase<ConvertFuncToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertFuncToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalOp<func::CallOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(&getContext());

  TypeConverter typeConverter;
  // Fallback for other types.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (!emitc::isSupportedEmitCType(type))
      return {};
    return type;
  });

  populateFuncToEmitCPatterns(typeConverter, patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
