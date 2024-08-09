//===- CIRGenOpenCL.cpp - OpenCL-specific logic for CIR generation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with OpenCL-specific logic of CIR generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

using namespace cir;
using namespace clang;

// Returns the address space id that should be produced to the
// kernel_arg_addr_space metadata. This is always fixed to the ids
// as specified in the SPIR 2.0 specification in order to differentiate
// for example in clGetKernelArgInfo() implementation between the address
// spaces with targets without unique mapping to the OpenCL address spaces
// (basically all single AS CPUs).
static unsigned ArgInfoAddressSpace(LangAS AS) {
  switch (AS) {
  case LangAS::opencl_global:
    return 1;
  case LangAS::opencl_constant:
    return 2;
  case LangAS::opencl_local:
    return 3;
  case LangAS::opencl_generic:
    return 4; // Not in SPIR 2.0 specs.
  case LangAS::opencl_global_device:
    return 5;
  case LangAS::opencl_global_host:
    return 6;
  default:
    return 0; // Assume private.
  }
}

void CIRGenModule::genKernelArgMetadata(mlir::cir::FuncOp Fn,
                                        const FunctionDecl *FD,
                                        CIRGenFunction *CGF) {
  assert(((FD && CGF) || (!FD && !CGF)) &&
         "Incorrect use - FD and CGF should either be both null or not!");
  // Create MDNodes that represent the kernel arg metadata.
  // Each MDNode is a list in the form of "key", N number of values which is
  // the same number of values as their are kernel arguments.

  const PrintingPolicy &Policy = getASTContext().getPrintingPolicy();

  // Integer values for the kernel argument address space qualifiers.
  SmallVector<int32_t, 8> addressQuals;

  // Attrs for the kernel argument access qualifiers (images only).
  SmallVector<mlir::Attribute, 8> accessQuals;

  // Attrs for the kernel argument type names.
  SmallVector<mlir::Attribute, 8> argTypeNames;

  // Attrs for the kernel argument base type names.
  SmallVector<mlir::Attribute, 8> argBaseTypeNames;

  // Attrs for the kernel argument type qualifiers.
  SmallVector<mlir::Attribute, 8> argTypeQuals;

  // Attrs for the kernel argument names.
  SmallVector<mlir::Attribute, 8> argNames;

  // OpenCL image and pipe types require special treatments for some metadata
  assert(!MissingFeatures::openCLBuiltinTypes());

  if (FD && CGF)
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      const ParmVarDecl *parm = FD->getParamDecl(i);
      // Get argument name.
      argNames.push_back(builder.getStringAttr(parm->getName()));

      if (!getLangOpts().OpenCL)
        continue;
      QualType ty = parm->getType();
      std::string typeQuals;

      // Get image and pipe access qualifier:
      if (ty->isImageType() || ty->isPipeType()) {
        llvm_unreachable("NYI");
      } else
        accessQuals.push_back(builder.getStringAttr("none"));

      auto getTypeSpelling = [&](QualType Ty) {
        auto typeName = Ty.getUnqualifiedType().getAsString(Policy);

        if (Ty.isCanonical()) {
          StringRef typeNameRef = typeName;
          // Turn "unsigned type" to "utype"
          if (typeNameRef.consume_front("unsigned "))
            return std::string("u") + typeNameRef.str();
          if (typeNameRef.consume_front("signed "))
            return typeNameRef.str();
        }

        return typeName;
      };

      if (ty->isPointerType()) {
        QualType pointeeTy = ty->getPointeeType();

        // Get address qualifier.
        addressQuals.push_back(
            ArgInfoAddressSpace(pointeeTy.getAddressSpace()));

        // Get argument type name.
        std::string typeName = getTypeSpelling(pointeeTy) + "*";
        std::string baseTypeName =
            getTypeSpelling(pointeeTy.getCanonicalType()) + "*";
        argTypeNames.push_back(builder.getStringAttr(typeName));
        argBaseTypeNames.push_back(builder.getStringAttr(baseTypeName));

        // Get argument type qualifiers:
        if (ty.isRestrictQualified())
          typeQuals = "restrict";
        if (pointeeTy.isConstQualified() ||
            (pointeeTy.getAddressSpace() == LangAS::opencl_constant))
          typeQuals += typeQuals.empty() ? "const" : " const";
        if (pointeeTy.isVolatileQualified())
          typeQuals += typeQuals.empty() ? "volatile" : " volatile";
      } else {
        uint32_t AddrSpc = 0;
        bool isPipe = ty->isPipeType();
        if (ty->isImageType() || isPipe)
          llvm_unreachable("NYI");

        addressQuals.push_back(AddrSpc);

        // Get argument type name.
        ty = isPipe ? ty->castAs<PipeType>()->getElementType() : ty;
        std::string typeName = getTypeSpelling(ty);
        std::string baseTypeName = getTypeSpelling(ty.getCanonicalType());

        // Remove access qualifiers on images
        // (as they are inseparable from type in clang implementation,
        // but OpenCL spec provides a special query to get access qualifier
        // via clGetKernelArgInfo with CL_KERNEL_ARG_ACCESS_QUALIFIER):
        if (ty->isImageType()) {
          llvm_unreachable("NYI");
        }

        argTypeNames.push_back(builder.getStringAttr(typeName));
        argBaseTypeNames.push_back(builder.getStringAttr(baseTypeName));

        if (isPipe)
          llvm_unreachable("NYI");
      }
      argTypeQuals.push_back(builder.getStringAttr(typeQuals));
    }

  bool shouldEmitArgName = getCodeGenOpts().EmitOpenCLArgMetadata ||
                           getCodeGenOpts().HIPSaveKernelArgName;

  if (getLangOpts().OpenCL) {
    // The kernel arg name is emitted only when `-cl-kernel-arg-info` is on,
    // since it is only used to support `clGetKernelArgInfo` which requires
    // `-cl-kernel-arg-info` to work. The other metadata are mandatory because
    // they are necessary for OpenCL runtime to set kernel argument.
    mlir::ArrayAttr resArgNames = {};
    if (shouldEmitArgName)
      resArgNames = builder.getArrayAttr(argNames);

    // Update the function's extra attributes with the kernel argument metadata.
    auto value = mlir::cir::OpenCLKernelArgMetadataAttr::get(
        Fn.getContext(), builder.getI32ArrayAttr(addressQuals),
        builder.getArrayAttr(accessQuals), builder.getArrayAttr(argTypeNames),
        builder.getArrayAttr(argBaseTypeNames),
        builder.getArrayAttr(argTypeQuals), resArgNames);
    mlir::NamedAttrList items{Fn.getExtraAttrs().getElements().getValue()};
    auto oldValue = items.set(value.getMnemonic(), value);
    if (oldValue != value) {
      Fn.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
          builder.getContext(), builder.getDictionaryAttr(items)));
    }
  } else {
    if (shouldEmitArgName)
      llvm_unreachable("NYI HIPSaveKernelArgName");
  }
}

void CIRGenFunction::buildKernelMetadata(const FunctionDecl *FD,
                                         mlir::cir::FuncOp Fn) {
  if (!FD->hasAttr<OpenCLKernelAttr>() && !FD->hasAttr<CUDAGlobalAttr>())
    return;

  CGM.genKernelArgMetadata(Fn, FD, this);

  if (!getLangOpts().OpenCL)
    return;

  using mlir::cir::OpenCLKernelMetadataAttr;

  mlir::ArrayAttr workGroupSizeHintAttr, reqdWorkGroupSizeAttr;
  mlir::TypeAttr vecTypeHintAttr;
  std::optional<bool> vecTypeHintSignedness;
  mlir::IntegerAttr intelReqdSubGroupSizeAttr;

  if (const VecTypeHintAttr *A = FD->getAttr<VecTypeHintAttr>()) {
    mlir::Type typeHintValue = getTypes().ConvertType(A->getTypeHint());
    vecTypeHintAttr = mlir::TypeAttr::get(typeHintValue);
    vecTypeHintSignedness =
        OpenCLKernelMetadataAttr::isSignedHint(typeHintValue);
  }

  if (const WorkGroupSizeHintAttr *A = FD->getAttr<WorkGroupSizeHintAttr>()) {
    workGroupSizeHintAttr = builder.getI32ArrayAttr({
        static_cast<int32_t>(A->getXDim()),
        static_cast<int32_t>(A->getYDim()),
        static_cast<int32_t>(A->getZDim()),
    });
  }

  if (const ReqdWorkGroupSizeAttr *A = FD->getAttr<ReqdWorkGroupSizeAttr>()) {
    reqdWorkGroupSizeAttr = builder.getI32ArrayAttr({
        static_cast<int32_t>(A->getXDim()),
        static_cast<int32_t>(A->getYDim()),
        static_cast<int32_t>(A->getZDim()),
    });
  }

  if (const OpenCLIntelReqdSubGroupSizeAttr *A =
          FD->getAttr<OpenCLIntelReqdSubGroupSizeAttr>()) {
    intelReqdSubGroupSizeAttr = builder.getI32IntegerAttr(A->getSubGroupSize());
  }

  // Skip the metadata attr if no hints are present.
  if (!vecTypeHintAttr && !workGroupSizeHintAttr && !reqdWorkGroupSizeAttr &&
      !intelReqdSubGroupSizeAttr)
    return;

  // Append the kernel metadata to the extra attributes dictionary.
  mlir::NamedAttrList attrs;
  attrs.append(Fn.getExtraAttrs().getElements());

  auto kernelMetadataAttr = OpenCLKernelMetadataAttr::get(
      builder.getContext(), workGroupSizeHintAttr, reqdWorkGroupSizeAttr,
      vecTypeHintAttr, vecTypeHintSignedness, intelReqdSubGroupSizeAttr);
  attrs.append(kernelMetadataAttr.getMnemonic(), kernelMetadataAttr);

  Fn.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
      builder.getContext(), attrs.getDictionary(builder.getContext())));
}

void CIRGenModule::buildOpenCLMetadata() {
  // SPIR v2.0 s2.13 - The OpenCL version used by the module is stored in the
  // opencl.ocl.version named metadata node.
  // C++ for OpenCL has a distinct mapping for versions compatibile with OpenCL.
  unsigned version = langOpts.getOpenCLCompatibleVersion();
  unsigned major = version / 100;
  unsigned minor = (version % 100) / 10;

  auto clVersionAttr =
      mlir::cir::OpenCLVersionAttr::get(builder.getContext(), major, minor);

  theModule->setAttr("cir.cl.version", clVersionAttr);
}
