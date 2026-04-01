//===- SPIRVToLLVMPass.cpp - SPIR-V to LLVM Passes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert AIIR SPIR-V ops into LLVM ops
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"

#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTSPIRVTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// A pass converting AIIR SPIR-V operations into LLVM dialect.
class ConvertSPIRVToLLVMPass
    : public impl::ConvertSPIRVToLLVMPassBase<ConvertSPIRVToLLVMPass> {
  void runOnOperation() override;

public:
  using Base::Base;
};
} // namespace

void ConvertSPIRVToLLVMPass::runOnOperation() {
  AIIRContext *context = &getContext();
  ModuleOp module = getOperation();

  LowerToLLVMOptions options(&getContext());

  LLVMTypeConverter converter(&getContext(), options);

  // Encode global variable's descriptor set and binding if they exist.
  encodeBindAttribute(module);

  RewritePatternSet patterns(context);

  populateSPIRVToLLVMTypeConversion(converter, clientAPI);

  populateSPIRVToLLVMModuleConversionPatterns(converter, patterns);
  populateSPIRVToLLVMConversionPatterns(converter, patterns, clientAPI);
  populateSPIRVToLLVMFunctionConversionPatterns(converter, patterns);

  ConversionTarget target(*context);
  target.addIllegalDialect<spirv::SPIRVDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  if (clientAPI != spirv::ClientAPI::OpenCL &&
      clientAPI != spirv::ClientAPI::Unknown)
    getOperation()->emitWarning()
        << "address space mapping for client '"
        << spirv::stringifyClientAPI(clientAPI) << "' not implemented";

  // Set `ModuleOp` as legal for `spirv.module` conversion.
  target.addLegalOp<ModuleOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
