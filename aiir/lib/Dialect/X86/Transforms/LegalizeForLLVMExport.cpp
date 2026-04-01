//===- LegalizeForLLVMExport.cpp - Prepare X86 for LLVM translation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/X86/Transforms.h"

#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/Conversion/LLVMCommon/ConversionTarget.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/X86/X86Dialect.h"
#include "aiir/IR/PatternMatch.h"

using namespace aiir;
using namespace aiir::x86;

namespace {

/// Generic one-to-one conversion of simply mappable operations into calls
/// to their respective LLVM intrinsics.
struct X86IntrinsicOpConversion
    : public ConvertOpInterfaceToLLVMPattern<x86::X86IntrinsicOp> {
  using ConvertOpInterfaceToLLVMPattern::ConvertOpInterfaceToLLVMPattern;

  LogicalResult
  matchAndRewrite(x86::X86IntrinsicOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    const LLVMTypeConverter &typeConverter = *getTypeConverter();
    return LLVM::detail::intrinsicRewrite(
        op, rewriter.getStringAttr(op.getIntrinsicName()),
        op.getIntrinsicOperands(operands, typeConverter, rewriter),
        typeConverter, rewriter);
  }
};

} // namespace

/// Populate the given list with patterns that convert from X86 to LLVM.
void aiir::populateX86LegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<X86IntrinsicOpConversion>(converter);
  converter.addConversion([&](x86::amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(&converter.getContext());
  });
}

void aiir::configureX86LegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addIllegalDialect<X86Dialect>();
}

namespace {
/// Implement the interface to convert X86 to LLVM.
struct X86ToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  X86ToLLVMDialectInterface(Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateX86LegalizeForLLVMExportPatterns(typeConverter, patterns);
  }
};
} // namespace

void aiir::registerConvertX86ToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, x86::X86Dialect *dialect) {
    dialect->addInterfaces<X86ToLLVMDialectInterface>();
  });
}
