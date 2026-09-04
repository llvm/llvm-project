//===- LowerToLLVMOpenCLMetadata.h - OpenCL metadata lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOPENCLMETADATA_H
#define CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOPENCLMETADATA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/SmallVector.h"

namespace cir {
namespace direct {

class OpenCLFunctionMetadataLowering {
public:
  OpenCLFunctionMetadataLowering(mlir::MLIRContext *ctx);

  bool lower(mlir::NamedAttribute attr, bool includeFunctionOnlyAttrs);
  void appendAttrs(llvm::SmallVectorImpl<mlir::NamedAttribute> &result) const;

private:
  void lower(cir::OpenCLKernelArgMetadataAttr clArgMetadata);

  mlir::MLIRContext *ctx;
  llvm::SmallVector<mlir::Attribute> functionMetadata;
};

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOPENCLMETADATA_H
