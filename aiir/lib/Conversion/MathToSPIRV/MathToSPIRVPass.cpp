//===- MathToSPIRVPass.cpp - Math to SPIR-V Passes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert standard dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/MathToSPIRV/MathToSPIRVPass.h"

#include "aiir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTMATHTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// A pass converting AIIR Math operations into the SPIR-V dialect.
class ConvertMathToSPIRVPass
    : public impl::ConvertMathToSPIRVPassBase<ConvertMathToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMathToSPIRVPass::runOnOperation() {
  AIIRContext *context = &getContext();
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);

  // Use UnrealizedConversionCast as the bridge so that we don't need to pull
  // in patterns for other dialects.
  target->addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(context);
  populateMathToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}
