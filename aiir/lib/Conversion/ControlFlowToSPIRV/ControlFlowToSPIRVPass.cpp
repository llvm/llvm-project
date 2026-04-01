//===- ControlFlowToSPIRVPass.cpp - ControlFlow to SPIR-V Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert ControlFlow dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"

#include "aiir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTCONTROLFLOWTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// A pass converting AIIR ControlFlow operations into the SPIR-V dialect.
class ConvertControlFlowToSPIRVPass final
    : public impl::ConvertControlFlowToSPIRVPassBase<
          ConvertControlFlowToSPIRVPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void ConvertControlFlowToSPIRVPass::runOnOperation() {
  AIIRContext *context = &getContext();
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVConversionOptions options;
  options.emulateLT32BitScalarTypes = this->emulateLT32BitScalarTypes;
  options.emulateUnsupportedFloatTypes = this->emulateUnsupportedFloatTypes;
  SPIRVTypeConverter typeConverter(targetAttr, options);

  // TODO: We should also take care of block argument type conversion.

  RewritePatternSet patterns(context);
  cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}
