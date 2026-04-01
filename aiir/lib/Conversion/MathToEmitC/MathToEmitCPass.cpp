//===- MathToEmitCPass.cpp - Math to EmitC Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Math dialect to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/MathToEmitC/MathToEmitCPass.h"
#include "aiir/Conversion/MathToEmitC/MathToEmitC.h"
#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTMATHTOEMITC
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
namespace {

//  Replaces Math operations with `emitc.call_opaque` operations.
struct ConvertMathToEmitC
    : public impl::ConvertMathToEmitCBase<ConvertMathToEmitC> {
  using ConvertMathToEmitCBase::ConvertMathToEmitCBase;

public:
  void runOnOperation() final;
};

} // namespace

void ConvertMathToEmitC::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalOp<emitc::CallOpaqueOp>();

  target.addIllegalOp<math::FloorOp, math::ExpOp, math::RoundOp, math::CosOp,
                      math::SinOp, math::Atan2Op, math::CeilOp, math::AcosOp,
                      math::AsinOp, math::AbsFOp, math::PowFOp>();

  RewritePatternSet patterns(&getContext());
  populateConvertMathToEmitCPatterns(patterns, languageTarget);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
