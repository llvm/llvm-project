//===- LowerToLLVMOpenCLMetadata.cpp - OpenCL metadata lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVMOpenCLMetadata.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {
namespace direct {

namespace {
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

using KernelArgStringMetadataGetter =
    mlir::ArrayAttr (cir::OpenCLKernelArgMetadataAttr::*)() const;

struct KernelArgStringMetadataMapping {
  llvm::StringLiteral metadataName;
  KernelArgStringMetadataGetter getMetadata;
  bool optional;
};

static unsigned getOpenCLArgInfoAddressSpace(cir::LangAddressSpace as) {
  switch (as) {
  case cir::LangAddressSpace::Default:
  case cir::LangAddressSpace::OffloadPrivate:
    return 0;
  case cir::LangAddressSpace::OffloadGlobal:
    return 1;
  case cir::LangAddressSpace::OffloadConstant:
    return 2;
  case cir::LangAddressSpace::OffloadLocal:
    return 3;
  case cir::LangAddressSpace::OffloadGeneric:
    return 4;
  case cir::LangAddressSpace::OffloadGlobalDevice:
    return 5;
  case cir::LangAddressSpace::OffloadGlobalHost:
    return 6;
  }
  llvm_unreachable("unknown CIR language address space");
}

static mlir::LLVM::MDNodeAttr
getAddrSpaceMetadataNode(cir::OpenCLKernelArgMetadataAttr clArgMetadata,
                         const LLVMMetadataNodeBuilder &metadataBuilder) {
  llvm::SmallVector<unsigned> addrSpaces;
  for (cir::LangAddressSpaceAttr addressSpace :
       clArgMetadata.getAddrSpace().getAsRange<cir::LangAddressSpaceAttr>())
    addrSpaces.push_back(getOpenCLArgInfoAddressSpace(addressSpace.getValue()));
  return metadataBuilder.getI32Node(addrSpaces);
}

static void addOpenCLKernelArgFunctionMetadata(
    mlir::MLIRContext *ctx, llvm::SmallVectorImpl<mlir::Attribute> &entries,
    llvm::StringRef name, mlir::LLVM::MDNodeAttr node) {
  entries.push_back(mlir::LLVM::FunctionMetadataAttr::get(
      ctx, mlir::StringAttr::get(ctx, name), node));
}

} // namespace

static void convertOpenCLKernelArgMetadata(
    cir::OpenCLKernelArgMetadataAttr clArgMetadata,
    llvm::SmallVectorImpl<mlir::Attribute> &entries) {
  mlir::MLIRContext *ctx = clArgMetadata.getContext();
  LLVMMetadataNodeBuilder metadataBuilder(ctx);

  addOpenCLKernelArgFunctionMetadata(
      ctx, entries, "kernel_arg_addr_space",
      getAddrSpaceMetadataNode(clArgMetadata, metadataBuilder));

  static constexpr KernelArgStringMetadataMapping stringMetadataMappings[] = {
      {"kernel_arg_access_qual",
       &cir::OpenCLKernelArgMetadataAttr::getAccessQual,
       /*optional=*/false},
      {"kernel_arg_type", &cir::OpenCLKernelArgMetadataAttr::getType,
       /*optional=*/false},
      {"kernel_arg_base_type", &cir::OpenCLKernelArgMetadataAttr::getBaseType,
       /*optional=*/false},
      {"kernel_arg_type_qual", &cir::OpenCLKernelArgMetadataAttr::getTypeQual,
       /*optional=*/false},
      {"kernel_arg_name", &cir::OpenCLKernelArgMetadataAttr::getName,
       /*optional=*/true},
  };

  for (const KernelArgStringMetadataMapping &mapping : stringMetadataMappings) {
    mlir::ArrayAttr metadata = (clArgMetadata.*mapping.getMetadata)();
    if (mapping.optional && !metadata)
      continue;
    addOpenCLKernelArgFunctionMetadata(ctx, entries, mapping.metadataName,
                                       metadataBuilder.getStringNode(metadata));
  }
}

OpenCLFunctionMetadataLowering::OpenCLFunctionMetadataLowering(
    mlir::MLIRContext *ctx)
    : ctx(ctx) {}

bool OpenCLFunctionMetadataLowering::lower(mlir::NamedAttribute attr,
                                           bool includeFunctionOnlyAttrs) {
  return llvm::TypeSwitch<mlir::Attribute, bool>(attr.getValue())
      .Case<cir::OpenCLKernelArgMetadataAttr>(
          [&](cir::OpenCLKernelArgMetadataAttr clArgMetadata) {
            if (!includeFunctionOnlyAttrs)
              return true;
            lower(clArgMetadata);
            return true;
          })
      .Default(false);
}

void OpenCLFunctionMetadataLowering::appendAttrs(
    llvm::SmallVectorImpl<mlir::NamedAttribute> &result) const {
  if (!functionMetadata.empty()) {
    result.push_back(mlir::NamedAttribute(
        mlir::LLVM::LLVMFuncOp::getFunctionMetadataAttrName(mlir::OperationName(
            mlir::LLVM::LLVMFuncOp::getOperationName(), ctx)),
        mlir::ArrayAttr::get(ctx, functionMetadata)));
  }
}

void OpenCLFunctionMetadataLowering::lower(
    cir::OpenCLKernelArgMetadataAttr clArgMetadata) {
  convertOpenCLKernelArgMetadata(clArgMetadata, functionMetadata);
}

} // namespace direct
} // namespace cir
