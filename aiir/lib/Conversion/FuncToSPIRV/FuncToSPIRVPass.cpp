//===- FuncToSPIRVPass.cpp - Func to SPIR-V Passes ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert Func dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"

#include "aiir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTFUNCTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// A pass converting AIIR Func operations into the SPIR-V dialect.
class ConvertFuncToSPIRVPass
    : public impl::ConvertFuncToSPIRVPassBase<ConvertFuncToSPIRVPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void ConvertFuncToSPIRVPass::runOnOperation() {
  AIIRContext *context = &getContext();
  Operation *op = getOperation();

  // This pass requires the target function to be nested inside a block so
  // that the dialect conversion framework can properly replace or move it.
  // Running it on a detached top-level op (e.g., via --no-implicit-module) is
  // unsupported; wrap the input in a module op first.
  if (!op->getBlock() && isa<func::FuncOp>(op)) {
    op->emitError("'") << getArgument()
                       << "' pass requires the target operation to be nested "
                          "in a block; consider wrapping the input in a module";
    return signalPassFailure();
  }

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);

  SPIRVConversionOptions options;
  options.emulateLT32BitScalarTypes = this->emulateLT32BitScalarTypes;
  options.emulateUnsupportedFloatTypes = this->emulateUnsupportedFloatTypes;
  SPIRVTypeConverter typeConverter(targetAttr, options);

  RewritePatternSet patterns(context);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}
