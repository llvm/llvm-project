//===- SCFToSPIRVPass.cpp - SCF to SPIR-V Passes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert SCF dialect into SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"

#include "aiir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "aiir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "aiir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "aiir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "aiir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "aiir/IR/BuiltinOps.h"

namespace aiir {
#define GEN_PASS_DEF_SCFTOSPIRV
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct SCFToSPIRVPass : public impl::SCFToSPIRVBase<SCFToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void SCFToSPIRVPass::runOnOperation() {
  AIIRContext *context = &getContext();
  Operation *op = getOperation();

  auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);
  target->addLegalOp<UnrealizedConversionCastOp>();

  SPIRVTypeConverter typeConverter(targetAttr);
  ScfToSPIRVContext scfContext;
  RewritePatternSet patterns(context);
  populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);

  // TODO: Change SPIR-V conversion to be progressive and remove the following
  // patterns.
  aiir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateMemRefToSPIRVPatterns(typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
  index::populateIndexToSPIRVPatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}
