//===- LowerToLLVMMetadata.cpp - CIR metadata lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVMMetadata.h"
#include "LowerToLLVMOpenCLMetadata.h"

#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/IR/OperationSupport.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace cir {
namespace direct {

FunctionMetadataLowering::FunctionMetadataLowering(
    mlir::Location loc, const mlir::TypeConverter &typeConverter)
    : loc(loc), typeConverter(typeConverter) {}

mlir::FailureOr<bool>
FunctionMetadataLowering::lower(mlir::NamedAttribute attr,
                                bool includeFunctionOnlyAttrs) {
  if (attr.getName() ==
      cir::CIRDialect::getOpenCLKernelExecutionMetadataAttrName()) {
    if (includeFunctionOnlyAttrs &&
        mlir::failed(lowerOpenCLKernelExecutionMetadata(
            mlir::cast<cir::OpenCLKernelExecutionMetadataAttr>(attr.getValue()),
            loc, typeConverter, attributes)))
      return mlir::failure();
    return true;
  }

  return llvm::TypeSwitch<mlir::Attribute, mlir::FailureOr<bool>>(
             attr.getValue())
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
      .Default(mlir::FailureOr<bool>(false));
}

void FunctionMetadataLowering::appendAttrs(
    llvm::SmallVectorImpl<mlir::NamedAttribute> &result) const {
  llvm::append_range(result, attributes);

  if (!functionMetadata.empty()) {
    mlir::MLIRContext *ctx = loc.getContext();
    result.push_back(mlir::NamedAttribute(
        mlir::LLVM::LLVMFuncOp::getFunctionMetadataAttrName(mlir::OperationName(
            mlir::LLVM::LLVMFuncOp::getOperationName(), ctx)),
        mlir::ArrayAttr::get(ctx, functionMetadata)));
  }
}

void FunctionMetadataLowering::lower(cir::MaxWorkGroupSizeAttr maxWGSize) {
  mlir::MLIRContext *ctx = loc.getContext();
  LLVMMetadataNodeBuilder metadataBuilder(ctx);
  unsigned sizes[] = {maxWGSize.getX(), maxWGSize.getY(), maxWGSize.getZ()};
  functionMetadata.push_back(mlir::LLVM::FunctionMetadataAttr::get(
      ctx, mlir::StringAttr::get(ctx, "max_work_group_size"),
      metadataBuilder.getI32Node(sizes)));
}

} // namespace direct
} // namespace cir
