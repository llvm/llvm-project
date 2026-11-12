//===- LowerToLLVMMetadata.cpp - CIR metadata lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVMMetadata.h"
#include "LowerToLLVMOpenCLMetadata.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/OperationSupport.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/TypeSwitch.h"

namespace cir {
namespace direct {

FunctionMetadataLowering::FunctionMetadataLowering(mlir::MLIRContext *ctx)
    : ctx(ctx) {}

bool FunctionMetadataLowering::lower(mlir::NamedAttribute attr,
                                     bool includeFunctionOnlyAttrs) {
  return llvm::TypeSwitch<mlir::Attribute, bool>(attr.getValue())
      .Case<cir::OpenCLKernelArgMetadataAttr>(
          [&](cir::OpenCLKernelArgMetadataAttr clArgMetadata) {
            if (includeFunctionOnlyAttrs)
              lowerOpenCLKernelArgMetadata(clArgMetadata, functionMetadata);
            return true;
          })
      .Case<cir::MaxWorkGroupSizeAttr>(
          [&](cir::MaxWorkGroupSizeAttr maxWGSize) {
            if (includeFunctionOnlyAttrs)
              lower(maxWGSize);
            return true;
          })
      .Default(false);
}

void FunctionMetadataLowering::appendAttrs(
    llvm::SmallVectorImpl<mlir::NamedAttribute> &result) const {
  if (!functionMetadata.empty()) {
    result.push_back(mlir::NamedAttribute(
        mlir::LLVM::LLVMFuncOp::getFunctionMetadataAttrName(mlir::OperationName(
            mlir::LLVM::LLVMFuncOp::getOperationName(), ctx)),
        mlir::ArrayAttr::get(ctx, functionMetadata)));
  }
}

void FunctionMetadataLowering::lower(cir::MaxWorkGroupSizeAttr maxWGSize) {
  LLVMMetadataNodeBuilder metadataBuilder(ctx);
  unsigned sizes[] = {maxWGSize.getX(), maxWGSize.getY(), maxWGSize.getZ()};
  functionMetadata.push_back(mlir::LLVM::FunctionMetadataAttr::get(
      ctx, mlir::StringAttr::get(ctx, "max_work_group_size"),
      metadataBuilder.getI32Node(sizes)));
}

} // namespace direct
} // namespace cir
