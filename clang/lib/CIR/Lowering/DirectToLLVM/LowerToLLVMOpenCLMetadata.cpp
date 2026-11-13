//===- LowerToLLVMOpenCLMetadata.cpp - OpenCL metadata lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVMOpenCLMetadata.h"
#include "LowerToLLVMMetadata.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {
namespace direct {

namespace {
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

void lowerOpenCLKernelArgMetadata(
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

static void createOpenCLVersionNamedMetadata(mlir::ModuleOp module,
                                             llvm::StringRef name,
                                             mlir::LLVM::MDNodeAttr node) {
  mlir::MLIRContext *ctx = module.getContext();
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());
  mlir::LLVM::NamedMetadataOp::create(builder, module.getLoc(), name,
                                      mlir::ArrayAttr::get(ctx, node));
}

static void emitOpenCLVersionMetadata(mlir::ModuleOp module,
                                      llvm::StringRef metadataName,
                                      mlir::Attribute versionAttr) {
  auto version = mlir::cast<cir::OpenCLVersionAttr>(versionAttr);
  mlir::MLIRContext *ctx = module.getContext();
  LLVMMetadataNodeBuilder metadataBuilder(ctx);

  unsigned versionMetadata[] = {
      static_cast<unsigned>(version.getMajorVersion()),
      static_cast<unsigned>(version.getMinorVersion())};
  createOpenCLVersionNamedMetadata(module, metadataName,
                                   metadataBuilder.getI32Node(versionMetadata));
}

void lowerOpenCLModuleMetadataAttrs(mlir::ModuleOp module) {
  struct OpenCLModuleMetadataMapping {
    llvm::StringRef cirAttrName;
    llvm::StringRef llvmMetadataName;
  };

  const OpenCLModuleMetadataMapping moduleMetadataMappings[] = {
      {cir::CIRDialect::getOpenCLVersionAttrName(), "opencl.ocl.version"},
      {cir::CIRDialect::getOpenCLCXXVersionAttrName(), "opencl.cxx.version"},
      {cir::CIRDialect::getOpenCLSPIRVersionAttrName(), "opencl.spir.version"},
  };

  for (const OpenCLModuleMetadataMapping &mapping : moduleMetadataMappings) {
    mlir::Attribute version = module->getAttr(mapping.cirAttrName);
    if (!version)
      continue;
    emitOpenCLVersionMetadata(module, mapping.llvmMetadataName, version);
    module->removeAttr(mapping.cirAttrName);
  }
}

} // namespace direct
} // namespace cir
