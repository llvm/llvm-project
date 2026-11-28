//===- LowerToLLVMMetadata.h - CIR metadata lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMMETADATA_H
#define CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMMETADATA_H

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace cir {
class MaxWorkGroupSizeAttr;

namespace direct {

class LLVMMetadataNodeBuilder {
public:
  explicit LLVMMetadataNodeBuilder(mlir::MLIRContext *ctx) : ctx(ctx) {}

  mlir::LLVM::MDConstantAttr getI32(unsigned value) const {
    mlir::IntegerType intTy = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerAttr intAttr = mlir::IntegerAttr::get(intTy, value);
    return mlir::LLVM::MDConstantAttr::get(ctx, intAttr);
  }

  mlir::LLVM::MDStringAttr getString(llvm::StringRef value) const {
    return mlir::LLVM::MDStringAttr::get(ctx,
                                         mlir::StringAttr::get(ctx, value));
  }

  mlir::LLVM::MDNodeAttr
  getNode(llvm::ArrayRef<mlir::Attribute> metadata) const {
    return mlir::LLVM::MDNodeAttr::get(ctx, metadata);
  }

  mlir::LLVM::MDNodeAttr getI32Node(llvm::ArrayRef<unsigned> values) const {
    llvm::SmallVector<mlir::Attribute> metadata;
    for (unsigned value : values)
      metadata.push_back(getI32(value));
    return getNode(metadata);
  }

  mlir::LLVM::MDNodeAttr getStringNode(mlir::ArrayAttr attrs) const {
    llvm::SmallVector<mlir::Attribute> metadata;
    for (mlir::StringAttr attr : attrs.getAsRange<mlir::StringAttr>())
      metadata.push_back(getString(attr.getValue()));
    return getNode(metadata);
  }

private:
  mlir::MLIRContext *ctx;
};

class FunctionMetadataLowering {
public:
  FunctionMetadataLowering(mlir::MLIRContext *ctx);

  bool lower(mlir::NamedAttribute attr, bool includeFunctionOnlyAttrs);
  void appendAttrs(llvm::SmallVectorImpl<mlir::NamedAttribute> &result) const;

private:
  void lower(cir::MaxWorkGroupSizeAttr maxWGSize);

  mlir::MLIRContext *ctx;
  llvm::SmallVector<mlir::Attribute> functionMetadata;
};

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERING_DIRECTTOLLVM_LOWERTOLLVMMETADATA_H
