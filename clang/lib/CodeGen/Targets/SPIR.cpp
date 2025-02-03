//===- SPIR.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// Base ABI and target codegen info implementation common between SPIR and
// SPIR-V.
//===----------------------------------------------------------------------===//

namespace {
class CommonSPIRABIInfo : public DefaultABIInfo {
public:
  CommonSPIRABIInfo(CodeGenTypes &CGT) : DefaultABIInfo(CGT) { setCCs(); }

private:
  void setCCs();
};

class SPIRVABIInfo : public CommonSPIRABIInfo {
public:
  SPIRVABIInfo(CodeGenTypes &CGT) : CommonSPIRABIInfo(CGT) {}
  void computeInfo(CGFunctionInfo &FI) const override;

private:
  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyKernelArgumentType(QualType Ty) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;
};
} // end anonymous namespace
namespace {
class CommonSPIRTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  CommonSPIRTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<CommonSPIRABIInfo>(CGT)) {}
  CommonSPIRTargetCodeGenInfo(std::unique_ptr<ABIInfo> ABIInfo)
      : TargetCodeGenInfo(std::move(ABIInfo)) {}

  LangAS getASTAllocaAddressSpace() const override {
    return getLangASFromTargetAS(
        getABIInfo().getDataLayout().getAllocaAddrSpace());
  }

  unsigned getOpenCLKernelCallingConv() const override;
  llvm::Type *getOpenCLType(CodeGenModule &CGM, const Type *T) const override;
  llvm::Type *getHLSLType(CodeGenModule &CGM, const Type *Ty) const override;
  llvm::Type *getSPIRVImageTypeFromHLSLResource(
      const HLSLAttributedResourceType::Attributes &attributes,
      llvm::Type *ElementType, llvm::LLVMContext &Ctx) const;
};
class SPIRVTargetCodeGenInfo : public CommonSPIRTargetCodeGenInfo {
public:
  SPIRVTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : CommonSPIRTargetCodeGenInfo(std::make_unique<SPIRVABIInfo>(CGT)) {}
  void setCUDAKernelCallingConvention(const FunctionType *&FT) const override;
  LangAS getGlobalVarAddressSpace(CodeGenModule &CGM,
                                  const VarDecl *D) const override;
  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const override;
  llvm::SyncScope::ID getLLVMSyncScopeID(const LangOptions &LangOpts,
                                         SyncScope Scope,
                                         llvm::AtomicOrdering Ordering,
                                         llvm::LLVMContext &Ctx) const override;
};

inline StringRef mapClangSyncScopeToLLVM(SyncScope Scope) {
  switch (Scope) {
  case SyncScope::HIPSingleThread:
  case SyncScope::SingleScope:
    return "singlethread";
  case SyncScope::HIPWavefront:
  case SyncScope::OpenCLSubGroup:
  case SyncScope::WavefrontScope:
    return "subgroup";
  case SyncScope::HIPWorkgroup:
  case SyncScope::OpenCLWorkGroup:
  case SyncScope::WorkgroupScope:
    return "workgroup";
  case SyncScope::HIPAgent:
  case SyncScope::OpenCLDevice:
  case SyncScope::DeviceScope:
    return "device";
  case SyncScope::SystemScope:
  case SyncScope::HIPSystem:
  case SyncScope::OpenCLAllSVMDevices:
    return "";
  }
  return "";
}
} // End anonymous namespace.

void CommonSPIRABIInfo::setCCs() {
  assert(getRuntimeCC() == llvm::CallingConv::C);
  RuntimeCC = llvm::CallingConv::SPIR_FUNC;
}

ABIArgInfo SPIRVABIInfo::classifyReturnType(QualType RetTy) const {
  if (getTarget().getTriple().getVendor() != llvm::Triple::AMD)
    return DefaultABIInfo::classifyReturnType(RetTy);
  if (!isAggregateTypeForABI(RetTy) || getRecordArgABI(RetTy, getCXXABI()))
    return DefaultABIInfo::classifyReturnType(RetTy);

  if (const RecordType *RT = RetTy->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    if (RD->hasFlexibleArrayMember())
      return DefaultABIInfo::classifyReturnType(RetTy);
  }

  // TODO: The AMDGPU ABI is non-trivial to represent in SPIR-V; in order to
  // avoid encoding various architecture specific bits here we return everything
  // as direct to retain type info for things like aggregates, for later perusal
  // when translating back to LLVM/lowering in the BE. This is also why we
  // disable flattening as the outcomes can mismatch between SPIR-V and AMDGPU.
  // This will be revisited / optimised in the future.
  return ABIArgInfo::getDirect(CGT.ConvertType(RetTy), 0u, nullptr, false);
}

ABIArgInfo SPIRVABIInfo::classifyKernelArgumentType(QualType Ty) const {
  if (getContext().getLangOpts().CUDAIsDevice) {
    // Coerce pointer arguments with default address space to CrossWorkGroup
    // pointers for HIPSPV/CUDASPV. When the language mode is HIP/CUDA, the
    // SPIRTargetInfo maps cuda_device to SPIR-V's CrossWorkGroup address space.
    llvm::Type *LTy = CGT.ConvertType(Ty);
    auto DefaultAS = getContext().getTargetAddressSpace(LangAS::Default);
    auto GlobalAS = getContext().getTargetAddressSpace(LangAS::cuda_device);
    auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(LTy);
    if (PtrTy && PtrTy->getAddressSpace() == DefaultAS) {
      LTy = llvm::PointerType::get(PtrTy->getContext(), GlobalAS);
      return ABIArgInfo::getDirect(LTy, 0, nullptr, false);
    }

    if (isAggregateTypeForABI(Ty)) {
      if (getTarget().getTriple().getVendor() == llvm::Triple::AMD)
        // TODO: The AMDGPU kernel ABI passes aggregates byref, which is not
        // currently expressible in SPIR-V; SPIR-V passes aggregates byval,
        // which the AMDGPU kernel ABI does not allow. Passing aggregates as
        // direct works around this impedance mismatch, as it retains type info
        // and can be correctly handled, post reverse-translation, by the AMDGPU
        // BE, which has to support this CC for legacy OpenCL purposes. It can
        // be brittle and does lead to performance degradation in certain
        // pathological cases. This will be revisited / optimised in the future,
        // once a way to deal with the byref/byval impedance mismatch is
        // identified.
        return ABIArgInfo::getDirect(LTy, 0, nullptr, false);
      // Force copying aggregate type in kernel arguments by value when
      // compiling CUDA targeting SPIR-V. This is required for the object
      // copied to be valid on the device.
      // This behavior follows the CUDA spec
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-function-argument-processing,
      // and matches the NVPTX implementation.
      return getNaturalAlignIndirect(Ty, /* byval */ true);
    }
  }
  return classifyArgumentType(Ty);
}

ABIArgInfo SPIRVABIInfo::classifyArgumentType(QualType Ty) const {
  if (getTarget().getTriple().getVendor() != llvm::Triple::AMD)
    return DefaultABIInfo::classifyArgumentType(Ty);
  if (!isAggregateTypeForABI(Ty))
    return DefaultABIInfo::classifyArgumentType(Ty);

  // Records with non-trivial destructors/copy-constructors should not be
  // passed by value.
  if (auto RAA = getRecordArgABI(Ty, getCXXABI()))
    return getNaturalAlignIndirect(Ty, RAA == CGCXXABI::RAA_DirectInMemory);

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    if (RD->hasFlexibleArrayMember())
      return DefaultABIInfo::classifyArgumentType(Ty);
  }

  return ABIArgInfo::getDirect(CGT.ConvertType(Ty), 0u, nullptr, false);
}

void SPIRVABIInfo::computeInfo(CGFunctionInfo &FI) const {
  // The logic is same as in DefaultABIInfo with an exception on the kernel
  // arguments handling.
  llvm::CallingConv::ID CC = FI.getCallingConvention();

  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  for (auto &I : FI.arguments()) {
    if (CC == llvm::CallingConv::SPIR_KERNEL) {
      I.info = classifyKernelArgumentType(I.type);
    } else {
      I.info = classifyArgumentType(I.type);
    }
  }
}

namespace clang {
namespace CodeGen {
void computeSPIRKernelABIInfo(CodeGenModule &CGM, CGFunctionInfo &FI) {
  if (CGM.getTarget().getTriple().isSPIRV())
    SPIRVABIInfo(CGM.getTypes()).computeInfo(FI);
  else
    CommonSPIRABIInfo(CGM.getTypes()).computeInfo(FI);
}
}
}

unsigned CommonSPIRTargetCodeGenInfo::getOpenCLKernelCallingConv() const {
  return llvm::CallingConv::SPIR_KERNEL;
}

void SPIRVTargetCodeGenInfo::setCUDAKernelCallingConvention(
    const FunctionType *&FT) const {
  // Convert HIP kernels to SPIR-V kernels.
  if (getABIInfo().getContext().getLangOpts().HIP) {
    FT = getABIInfo().getContext().adjustFunctionType(
        FT, FT->getExtInfo().withCallingConv(CC_OpenCLKernel));
    return;
  }
}

LangAS
SPIRVTargetCodeGenInfo::getGlobalVarAddressSpace(CodeGenModule &CGM,
                                                 const VarDecl *D) const {
  assert(!CGM.getLangOpts().OpenCL &&
         !(CGM.getLangOpts().CUDA && CGM.getLangOpts().CUDAIsDevice) &&
         "Address space agnostic languages only");
  // If we're here it means that we're using the SPIRDefIsGen ASMap, hence for
  // the global AS we can rely on either cuda_device or sycl_global to be
  // correct; however, since this is not a CUDA Device context, we use
  // sycl_global to prevent confusion with the assertion.
  LangAS DefaultGlobalAS = getLangASFromTargetAS(
      CGM.getContext().getTargetAddressSpace(LangAS::sycl_global));
  if (!D)
    return DefaultGlobalAS;

  LangAS AddrSpace = D->getType().getAddressSpace();
  if (AddrSpace != LangAS::Default)
    return AddrSpace;

  return DefaultGlobalAS;
}

void SPIRVTargetCodeGenInfo::setTargetAttributes(
    const Decl *D, llvm::GlobalValue *GV, CodeGen::CodeGenModule &M) const {
  if (!M.getLangOpts().HIP ||
      M.getTarget().getTriple().getVendor() != llvm::Triple::AMD)
    return;
  if (GV->isDeclaration())
    return;

  auto F = dyn_cast<llvm::Function>(GV);
  if (!F)
    return;

  auto FD = dyn_cast_or_null<FunctionDecl>(D);
  if (!FD)
    return;
  if (!FD->hasAttr<CUDAGlobalAttr>())
    return;

  unsigned N = M.getLangOpts().GPUMaxThreadsPerBlock;
  if (auto FlatWGS = FD->getAttr<AMDGPUFlatWorkGroupSizeAttr>())
    N = FlatWGS->getMax()->EvaluateKnownConstInt(M.getContext()).getExtValue();

  // We encode the maximum flat WG size in the first component of the 3D
  // max_work_group_size attribute, which will get reverse translated into the
  // original AMDGPU attribute when targeting AMDGPU.
  auto Int32Ty = llvm::IntegerType::getInt32Ty(M.getLLVMContext());
  llvm::Metadata *AttrMDArgs[] = {
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, N)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 1)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, 1))};

  F->setMetadata("max_work_group_size",
                 llvm::MDNode::get(M.getLLVMContext(), AttrMDArgs));
}

llvm::SyncScope::ID
SPIRVTargetCodeGenInfo::getLLVMSyncScopeID(const LangOptions &, SyncScope Scope,
                                           llvm::AtomicOrdering,
                                           llvm::LLVMContext &Ctx) const {
  return Ctx.getOrInsertSyncScopeID(mapClangSyncScopeToLLVM(Scope));
}

/// Construct a SPIR-V target extension type for the given OpenCL image type.
static llvm::Type *getSPIRVImageType(llvm::LLVMContext &Ctx, StringRef BaseType,
                                     StringRef OpenCLName,
                                     unsigned AccessQualifier) {
  // These parameters compare to the operands of OpTypeImage (see
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage
  // for more details). The first 6 integer parameters all default to 0, and
  // will be changed to 1 only for the image type(s) that set the parameter to
  // one. The 7th integer parameter is the access qualifier, which is tacked on
  // at the end.
  SmallVector<unsigned, 7> IntParams = {0, 0, 0, 0, 0, 0};

  // Choose the dimension of the image--this corresponds to the Dim enum in
  // SPIR-V (first integer parameter of OpTypeImage).
  if (OpenCLName.starts_with("image2d"))
    IntParams[0] = 1; // 1D
  else if (OpenCLName.starts_with("image3d"))
    IntParams[0] = 2; // 2D
  else if (OpenCLName == "image1d_buffer")
    IntParams[0] = 5; // Buffer
  else
    assert(OpenCLName.starts_with("image1d") && "Unknown image type");

  // Set the other integer parameters of OpTypeImage if necessary. Note that the
  // OpenCL image types don't provide any information for the Sampled or
  // Image Format parameters.
  if (OpenCLName.contains("_depth"))
    IntParams[1] = 1;
  if (OpenCLName.contains("_array"))
    IntParams[2] = 1;
  if (OpenCLName.contains("_msaa"))
    IntParams[3] = 1;

  // Access qualifier
  IntParams.push_back(AccessQualifier);

  return llvm::TargetExtType::get(Ctx, BaseType, {llvm::Type::getVoidTy(Ctx)},
                                  IntParams);
}

llvm::Type *CommonSPIRTargetCodeGenInfo::getOpenCLType(CodeGenModule &CGM,
                                                       const Type *Ty) const {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  if (auto *PipeTy = dyn_cast<PipeType>(Ty))
    return llvm::TargetExtType::get(Ctx, "spirv.Pipe", {},
                                    {!PipeTy->isReadOnly()});
  if (auto *BuiltinTy = dyn_cast<BuiltinType>(Ty)) {
    enum AccessQualifier : unsigned { AQ_ro = 0, AQ_wo = 1, AQ_rw = 2 };
    switch (BuiltinTy->getKind()) {
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
    case BuiltinType::Id:                                                      \
      return getSPIRVImageType(Ctx, "spirv.Image", #ImgType, AQ_##Suffix);
#include "clang/Basic/OpenCLImageTypes.def"
    case BuiltinType::OCLSampler:
      return llvm::TargetExtType::get(Ctx, "spirv.Sampler");
    case BuiltinType::OCLEvent:
      return llvm::TargetExtType::get(Ctx, "spirv.Event");
    case BuiltinType::OCLClkEvent:
      return llvm::TargetExtType::get(Ctx, "spirv.DeviceEvent");
    case BuiltinType::OCLQueue:
      return llvm::TargetExtType::get(Ctx, "spirv.Queue");
    case BuiltinType::OCLReserveID:
      return llvm::TargetExtType::get(Ctx, "spirv.ReserveId");
#define INTEL_SUBGROUP_AVC_TYPE(Name, Id)                                      \
    case BuiltinType::OCLIntelSubgroupAVC##Id:                                 \
      return llvm::TargetExtType::get(Ctx, "spirv.Avc" #Id "INTEL");
#include "clang/Basic/OpenCLExtensionTypes.def"
    default:
      return nullptr;
    }
  }

  return nullptr;
}

llvm::Type *CommonSPIRTargetCodeGenInfo::getHLSLType(CodeGenModule &CGM,
                                                     const Type *Ty) const {
  auto *ResType = dyn_cast<HLSLAttributedResourceType>(Ty);
  if (!ResType)
    return nullptr;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  const HLSLAttributedResourceType::Attributes &ResAttrs = ResType->getAttrs();
  switch (ResAttrs.ResourceClass) {
  case llvm::dxil::ResourceClass::UAV:
  case llvm::dxil::ResourceClass::SRV: {
    // TypedBuffer and RawBuffer both need element type
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull())
      return nullptr;

    assert(!ResAttrs.RawBuffer &&
           "Raw buffers handles are not implemented for SPIR-V yet");
    assert(!ResAttrs.IsROV &&
           "Rasterizer order views not implemented for SPIR-V yet");

    // convert element type
    llvm::Type *ElemType = CGM.getTypes().ConvertType(ContainedTy);
    return getSPIRVImageTypeFromHLSLResource(ResAttrs, ElemType, Ctx);
  }
  case llvm::dxil::ResourceClass::CBuffer:
    llvm_unreachable("CBuffer handles are not implemented for SPIR-V yet");
    break;
  case llvm::dxil::ResourceClass::Sampler:
    return llvm::TargetExtType::get(Ctx, "spirv.Sampler");
  }
  return nullptr;
}

llvm::Type *CommonSPIRTargetCodeGenInfo::getSPIRVImageTypeFromHLSLResource(
    const HLSLAttributedResourceType::Attributes &attributes,
    llvm::Type *ElementType, llvm::LLVMContext &Ctx) const {

  if (ElementType->isVectorTy())
    ElementType = ElementType->getScalarType();

  assert((ElementType->isIntegerTy() || ElementType->isFloatingPointTy()) &&
         "The element type for a SPIR-V resource must be a scalar integer or "
         "floating point type.");

  // These parameters correspond to the operands to the OpTypeImage SPIR-V
  // instruction. See
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage.
  SmallVector<unsigned, 6> IntParams(6, 0);

  // Dim
  // For now we assume everything is a buffer.
  IntParams[0] = 5;

  // Depth
  // HLSL does not indicate if it is a depth texture or not, so we use unknown.
  IntParams[1] = 2;

  // Arrayed
  IntParams[2] = 0;

  // MS
  IntParams[3] = 0;

  // Sampled
  IntParams[4] =
      attributes.ResourceClass == llvm::dxil::ResourceClass::UAV ? 2 : 1;

  // Image format.
  // Setting to unknown for now.
  IntParams[5] = 0;

  return llvm::TargetExtType::get(Ctx, "spirv.Image", {ElementType}, IntParams);
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createCommonSPIRTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<CommonSPIRTargetCodeGenInfo>(CGM.getTypes());
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createSPIRVTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<SPIRVTargetCodeGenInfo>(CGM.getTypes());
}
