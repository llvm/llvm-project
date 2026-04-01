//===- MemRefToSPIRVPass.cpp - MemRef to SPIR-V Passes ----------------===//
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

#include "aiir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"

#include "aiir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTMEMREFTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// A pass converting AIIR MemRef operations into the SPIR-V dialect.
class ConvertMemRefToSPIRVPass
    : public impl::ConvertMemRefToSPIRVPassBase<ConvertMemRefToSPIRVPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void ConvertMemRefToSPIRVPass::runOnOperation() {
  AIIRContext *context = &getContext();
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVConversionOptions options;
  options.boolNumBits = this->boolNumBits;
  options.use64bitIndex = this->use64bitIndex;
  SPIRVTypeConverter typeConverter(targetAttr, options);

  // Use UnrealizedConversionCast as the bridge so that we don't need to pull in
  // patterns for other dialects.
  target->addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(context);
  populateMemRefToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}
