//===- ArithToEmitCPass.cpp - Arith to EmitC Pass ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Arith dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ArithToEmitC/ArithToEmitCPass.h"

#include "aiir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTARITHTOEMITC
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct ConvertArithToEmitC
    : public impl::ConvertArithToEmitCBase<ConvertArithToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertArithToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(&getContext());

  TypeConverter typeConverter;
  // Fallback for other types.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (!emitc::isSupportedEmitCType(type))
      return {};
    return type;
  });

  populateArithToEmitCPatterns(typeConverter, patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
