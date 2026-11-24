//===- LowerToLLVMOpenCLMetadata.h - OpenCL metadata lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOPENCLMETADATA_H
#define CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOPENCLMETADATA_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/SmallVector.h"

namespace cir {
namespace direct {

void lowerOpenCLKernelArgMetadata(
    cir::OpenCLKernelArgMetadataAttr clArgMetadata,
    llvm::SmallVectorImpl<mlir::Attribute> &entries);

void lowerOpenCLModuleMetadataAttrs(mlir::ModuleOp module);

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMOPENCLMETADATA_H
