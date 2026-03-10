//===- SPIR.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "HLSLBufferLayoutBuilder.h"
#include "TargetInfo.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/IR/DerivedTypes.h"

#include <stdint.h>
#include <utility>

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
  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;

private:
  ABIArgInfo classifyKernelArgumentType(QualType Ty) const;
};

class AMDGCNSPIRVABIInfo : public SPIRVABIInfo {
  // TODO: this should be unified / shared with AMDGPU, ideally we'd like to
  //       re-use AMDGPUABIInfo eventually, rather than duplicate.
  static constexpr unsigned MaxNumRegsForArgsRet = 16; // 16 32-bit registers
  mutable unsigned NumRegsLeft = 0;

  uint64_t numRegsForType(QualType Ty) const;

  bool isHomogeneousAggregateBaseType(QualType Ty) const override {
    return true;
  }
  bool isHomogeneousAggregateSmallEnough(const Type *Base,
                                         uint64_t Members) const override {
    uint32_t NumRegs = (getContext().getTypeSize(Base) + 31) / 32;

    // Homogeneous Aggregates may occupy at most 16 registers.
    return Members * NumRegs <= MaxNumRegsForArgsRet;
  }

  // Coerce HIP scalar pointer arguments from generic pointers to global ones.
  llvm::Type *coerceKernelArgumentType(llvm::Type *Ty, unsigned FromAS,
                                       unsigned ToAS) const;

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyKernelArgumentType(QualType Ty) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

public:
  AMDGCNSPIRVABIInfo(CodeGenTypes &CGT) : SPIRVABIInfo(CGT) {}
  void computeInfo(CGFunctionInfo &FI) const override;

  llvm::FixedVectorType *
  getOptimalVectorMemoryType(llvm::FixedVectorType *Ty,
                             const LangOptions &LangOpt) const override;
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

  unsigned getDeviceKernelCallingConv() const override;
  llvm::Type *getOpenCLType(CodeGenModule &CGM, const Type *T) const override;
  llvm::Type *getHLSLType(CodeGenModule &CGM, const Type *Ty,
                          const CGHLSLOffsetInfo &OffsetInfo) const override;

  llvm::Type *getHLSLPadding(CodeGenModule &CGM,
                             CharUnits NumBytes) const override {
    unsigned Size = NumBytes.getQuantity();
    return llvm::TargetExtType::get(CGM.getLLVMContext(), "spirv.Padding", {},
                                    {Size});
  }

  bool isHLSLPadding(llvm::Type *Ty) const override {
    if (auto *TET = dyn_cast<llvm::TargetExtType>(Ty))
      return TET->getName() == "spirv.Padding";
    return false;
  }

  llvm::Type *getSPIRVImageTypeFromHLSLResource(
      const HLSLAttributedResourceType::Attributes &attributes,
      QualType SampledType, CodeGenModule &CGM) const;
  void
  setOCLKernelStubCallingConvention(const FunctionType *&FT) const override;
  llvm::Constant *getNullPointer(const CodeGen::CodeGenModule &CGM,
                                 llvm::PointerType *T,
                                 QualType QT) const override;
};
class SPIRVTargetCodeGenInfo : public CommonSPIRTargetCodeGenInfo {
public:
  SPIRVTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : CommonSPIRTargetCodeGenInfo(
            (CGT.getTarget().getTriple().getVendor() == llvm::Triple::AMD)
                ? std::make_unique<AMDGCNSPIRVABIInfo>(CGT)
                : std::make_unique<SPIRVABIInfo>(CGT)) {}
  void setCUDAKernelCallingConvention(const FunctionType *&FT) const override;
  LangAS getGlobalVarAddressSpace(CodeGenModule &CGM,
                                  const VarDecl *D) const override;
  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const override;
  StringRef getLLVMSyncScopeStr(const LangOptions &LangOpts, SyncScope Scope,
                                llvm::AtomicOrdering Ordering) const override;
  bool supportsLibCall() const override {
    return getABIInfo().getTarget().getTriple().getVendor() !=
           llvm::Triple::AMD;
  }
};
} // End anonymous namespace.

void CommonSPIRABIInfo::setCCs() {
  assert(getRuntimeCC() == llvm::CallingConv::C);
  RuntimeCC = llvm::CallingConv::SPIR_FUNC;
}

ABIArgInfo SPIRVABIInfo::classifyKernelArgumentType(QualType Ty) const {
  if (getContext().getLangOpts().isTargetDevice()) {
    // Coerce pointer arguments with default address space to CrossWorkGroup
    // pointers for target devices as default address space kernel arguments
    // are not allowed. We use the opencl_global language address space which
    // always maps to CrossWorkGroup.
    llvm::Type *LTy = CGT.ConvertType(Ty);
    auto DefaultAS = getContext().getTargetAddressSpace(LangAS::Default);
    auto GlobalAS = getContext().getTargetAddressSpace(LangAS::opencl_global);
    auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(LTy);
    if (PtrTy && PtrTy->getAddressSpace() == DefaultAS) {
      LTy = llvm::PointerType::get(PtrTy->getContext(), GlobalAS);
      return ABIArgInfo::getDirect(LTy, 0, nullptr, false);
    }

    if (isAggregateTypeForABI(Ty)) {
      // Force copying aggregate type in kernel arguments by value when
      // compiling CUDA targeting SPIR-V. This is required for the object
      // copied to be valid on the device.
      // This behavior follows the CUDA spec
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-function-argument-processing,
      // and matches the NVPTX implementation. TODO: hardcoding to 0 should be
      // revisited if HIPSPV / byval starts making use of the AS of an indirect
      // arg.
      return getNaturalAlignIndirect(Ty, /*AddrSpace=*/0, /*byval=*/true);
    }
  }
  return classifyArgumentType(Ty);
}

void SPIRVABIInfo::computeInfo(CGFunctionInfo &FI) const {
  // The logic is same as in DefaultABIInfo with an exception on the kernel
  // arguments handling.
  llvm::CallingConv::ID CC = FI.getCallingConvention();

  for (auto &&[ArgumentsCount, I] : llvm::enumerate(FI.arguments()))
    I.info = ArgumentsCount < FI.getNumRequiredArgs()
                 ? classifyArgumentType(I.type)
                 : ABIArgInfo::getDirect();

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

RValue SPIRVABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                               QualType Ty, AggValueSlot Slot) const {
  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, /*IsIndirect=*/false,
                          getContext().getTypeInfoInChars(Ty),
                          CharUnits::fromQuantity(1),
                          /*AllowHigherAlign=*/true, Slot);
}

uint64_t AMDGCNSPIRVABIInfo::numRegsForType(QualType Ty) const {
  // This duplicates the AMDGPUABI computation.
  uint64_t NumRegs = 0;

  if (const VectorType *VT = Ty->getAs<VectorType>()) {
    // Compute from the number of elements. The reported size is based on the
    // in-memory size, which includes the padding 4th element for 3-vectors.
    QualType EltTy = VT->getElementType();
    uint64_t EltSize = getContext().getTypeSize(EltTy);

    // 16-bit element vectors should be passed as packed.
    if (EltSize == 16)
      return (VT->getNumElements() + 1) / 2;

    uint64_t EltNumRegs = (EltSize + 31) / 32;
    return EltNumRegs * VT->getNumElements();
  }

  if (const auto *RD = Ty->getAsRecordDecl()) {
    assert(!RD->hasFlexibleArrayMember());

    for (const FieldDecl *Field : RD->fields()) {
      QualType FieldTy = Field->getType();
      NumRegs += numRegsForType(FieldTy);
    }

    return NumRegs;
  }

  return (getContext().getTypeSize(Ty) + 31) / 32;
}

llvm::Type *AMDGCNSPIRVABIInfo::coerceKernelArgumentType(llvm::Type *Ty,
                                                         unsigned FromAS,
                                                         unsigned ToAS) const {
  // Single value types.
  auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(Ty);
  if (PtrTy && PtrTy->getAddressSpace() == FromAS)
    return llvm::PointerType::get(Ty->getContext(), ToAS);
  return Ty;
}

ABIArgInfo AMDGCNSPIRVABIInfo::classifyReturnType(QualType RetTy) const {
  if (!isAggregateTypeForABI(RetTy) || getRecordArgABI(RetTy, getCXXABI()))
    return DefaultABIInfo::classifyReturnType(RetTy);

  // Ignore empty structs/unions.
  if (isEmptyRecord(getContext(), RetTy, true))
    return ABIArgInfo::getIgnore();

  // Lower single-element structs to just return a regular value.
  if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
    return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

  if (const auto *RD = RetTy->getAsRecordDecl();
      RD && RD->hasFlexibleArrayMember())
    return DefaultABIInfo::classifyReturnType(RetTy);

  // Pack aggregates <= 4 bytes into single VGPR or pair.
  uint64_t Size = getContext().getTypeSize(RetTy);
  if (Size <= 16)
    return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));

  if (Size <= 32)
    return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

  // TODO: This carried over from AMDGPU oddity, we retain it to
  //       ensure consistency, but it might be reasonable to return Int64.
  if (Size <= 64) {
    llvm::Type *I32Ty = llvm::Type::getInt32Ty(getVMContext());
    return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
  }

  if (numRegsForType(RetTy) <= MaxNumRegsForArgsRet)
    return ABIArgInfo::getDirect();
  return DefaultABIInfo::classifyReturnType(RetTy);
}

/// For kernels all parameters are really passed in a special buffer. It doesn't
/// make sense to pass anything byval, so everything must be direct.
ABIArgInfo AMDGCNSPIRVABIInfo::classifyKernelArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  // TODO: Can we omit empty structs?

  if (const Type *SeltTy = isSingleElementStruct(Ty, getContext()))
    Ty = QualType(SeltTy, 0);

  llvm::Type *OrigLTy = CGT.ConvertType(Ty);
  llvm::Type *LTy = OrigLTy;
  if (getContext().getLangOpts().isTargetDevice()) {
    LTy = coerceKernelArgumentType(
        OrigLTy, /*FromAS=*/getContext().getTargetAddressSpace(LangAS::Default),
        /*ToAS=*/getContext().getTargetAddressSpace(LangAS::opencl_global));
  }

  // FIXME: This doesn't apply the optimization of coercing pointers in structs
  // to global address space when using byref. This would require implementing a
  // new kind of coercion of the in-memory type when for indirect arguments.
  if (LTy == OrigLTy && isAggregateTypeForABI(Ty)) {
    return ABIArgInfo::getIndirectAliased(
        getContext().getTypeAlignInChars(Ty),
        getContext().getTargetAddressSpace(LangAS::opencl_constant),
        false /*Realign*/, nullptr /*Padding*/);
  }

  // TODO: inhibiting flattening is an AMDGPU workaround for Clover, which might
  //       be vestigial and should be revisited.
  return ABIArgInfo::getDirect(LTy, 0, nullptr, false);
}

ABIArgInfo AMDGCNSPIRVABIInfo::classifyArgumentType(QualType Ty) const {
  assert(NumRegsLeft <= MaxNumRegsForArgsRet && "register estimate underflow");

  Ty = useFirstFieldIfTransparentUnion(Ty);

  // TODO: support for variadics.

  if (!isAggregateTypeForABI(Ty)) {
    ABIArgInfo ArgInfo = DefaultABIInfo::classifyArgumentType(Ty);
    if (!ArgInfo.isIndirect()) {
      uint64_t NumRegs = numRegsForType(Ty);
      NumRegsLeft -= std::min(NumRegs, uint64_t{NumRegsLeft});
    }

    return ArgInfo;
  }

  // Records with non-trivial destructors/copy-constructors should not be
  // passed by value.
  if (auto RAA = getRecordArgABI(Ty, getCXXABI()))
    return getNaturalAlignIndirect(Ty, getDataLayout().getAllocaAddrSpace(),
                                   RAA == CGCXXABI::RAA_DirectInMemory);

  // Ignore empty structs/unions.
  if (isEmptyRecord(getContext(), Ty, true))
    return ABIArgInfo::getIgnore();

  // Lower single-element structs to just pass a regular value. TODO: We
  // could do reasonable-size multiple-element structs too, using getExpand(),
  // though watch out for things like bitfields.
  if (const Type *SeltTy = isSingleElementStruct(Ty, getContext()))
    return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

  if (const auto *RD = Ty->getAsRecordDecl();
      RD && RD->hasFlexibleArrayMember())
    return DefaultABIInfo::classifyArgumentType(Ty);

  uint64_t Size = getContext().getTypeSize(Ty);
  if (Size <= 64) {
    // Pack aggregates <= 8 bytes into single VGPR or pair.
    unsigned NumRegs = (Size + 31) / 32;
    NumRegsLeft -= std::min(NumRegsLeft, NumRegs);

    if (Size <= 16)
      return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));

    if (Size <= 32)
      return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

    // TODO: This is an AMDGPU oddity, and might be vestigial, we retain it to
    //       ensure consistency, but it should be revisited.
    llvm::Type *I32Ty = llvm::Type::getInt32Ty(getVMContext());
    return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
  }

  if (NumRegsLeft > 0) {
    uint64_t NumRegs = numRegsForType(Ty);
    if (NumRegsLeft >= NumRegs) {
      NumRegsLeft -= NumRegs;
      return ABIArgInfo::getDirect();
    }
  }

  // Use pass-by-reference in stead of pass-by-value for struct arguments in
  // function ABI.
  return ABIArgInfo::getIndirectAliased(
      getContext().getTypeAlignInChars(Ty),
      getContext().getTargetAddressSpace(LangAS::opencl_private));
}

void AMDGCNSPIRVABIInfo::computeInfo(CGFunctionInfo &FI) const {
  llvm::CallingConv::ID CC = FI.getCallingConvention();

  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  NumRegsLeft = MaxNumRegsForArgsRet;
  for (auto &I : FI.arguments()) {
    if (CC == llvm::CallingConv::SPIR_KERNEL)
      I.info = classifyKernelArgumentType(I.type);
    else
      I.info = classifyArgumentType(I.type);
  }
}

llvm::FixedVectorType *AMDGCNSPIRVABIInfo::getOptimalVectorMemoryType(
    llvm::FixedVectorType *Ty, const LangOptions &LangOpt) const {
  // AMDGPU has legal instructions for 96-bit so 3x32 can be supported.
  if (Ty->getNumElements() == 3 && getDataLayout().getTypeSizeInBits(Ty) == 96)
    return Ty;
  return DefaultABIInfo::getOptimalVectorMemoryType(Ty, LangOpt);
}

namespace clang {
namespace CodeGen {
void computeSPIRKernelABIInfo(CodeGenModule &CGM, CGFunctionInfo &FI) {
  if (CGM.getTarget().getTriple().isSPIRV()) {
    if (CGM.getTarget().getTriple().getVendor() == llvm::Triple::AMD)
      AMDGCNSPIRVABIInfo(CGM.getTypes()).computeInfo(FI);
    else
      SPIRVABIInfo(CGM.getTypes()).computeInfo(FI);
  } else {
    CommonSPIRABIInfo(CGM.getTypes()).computeInfo(FI);
  }
}
}
}

unsigned CommonSPIRTargetCodeGenInfo::getDeviceKernelCallingConv() const {
  return llvm::CallingConv::SPIR_KERNEL;
}

void SPIRVTargetCodeGenInfo::setCUDAKernelCallingConvention(
    const FunctionType *&FT) const {
  // Convert HIP kernels to SPIR-V kernels.
  if (getABIInfo().getContext().getLangOpts().HIP) {
    FT = getABIInfo().getContext().adjustFunctionType(
        FT, FT->getExtInfo().withCallingConv(CC_DeviceKernel));
    return;
  }
}

void CommonSPIRTargetCodeGenInfo::setOCLKernelStubCallingConvention(
    const FunctionType *&FT) const {
  FT = getABIInfo().getContext().adjustFunctionType(
      FT, FT->getExtInfo().withCallingConv(CC_SpirFunction));
}

// LLVM currently assumes a null pointer has the bit pattern 0, but some GPU
// targets use a non-zero encoding for null in certain address spaces.
// Because SPIR(-V) is a generic target and the bit pattern of null in
// non-generic AS is unspecified, materialize null in non-generic AS via an
// addrspacecast from null in generic AS. This allows later lowering to
// substitute the target's real sentinel value.
llvm::Constant *
CommonSPIRTargetCodeGenInfo::getNullPointer(const CodeGen::CodeGenModule &CGM,
                                            llvm::PointerType *PT,
                                            QualType QT) const {
  LangAS AS = QT->getUnqualifiedDesugaredType()->isNullPtrType()
                  ? LangAS::Default
                  : QT->getPointeeType().getAddressSpace();
  unsigned ASAsInt = static_cast<unsigned>(AS);
  unsigned FirstTargetASAsInt =
      static_cast<unsigned>(LangAS::FirstTargetAddressSpace);
  unsigned CodeSectionINTELAS = FirstTargetASAsInt + 9;
  // As per SPV_INTEL_function_pointers, it is illegal to addrspacecast
  // function pointers to/from the generic AS.
  bool IsFunctionPtrAS =
      CGM.getTriple().isSPIRV() && ASAsInt == CodeSectionINTELAS;
  if (AS == LangAS::Default || AS == LangAS::opencl_generic ||
      AS == LangAS::opencl_constant || IsFunctionPtrAS)
    return llvm::ConstantPointerNull::get(PT);

  auto &Ctx = CGM.getContext();
  auto NPT = llvm::PointerType::get(
      PT->getContext(), Ctx.getTargetAddressSpace(LangAS::opencl_generic));
  return llvm::ConstantExpr::getAddrSpaceCast(
      llvm::ConstantPointerNull::get(NPT), PT);
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
  if (GV->isDeclaration())
    return;

  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D);
  if (!FD)
    return;

  llvm::Function *F = dyn_cast<llvm::Function>(GV);
  assert(F && "Expected GlobalValue to be a Function");

  if (!M.getLangOpts().HIP ||
      M.getTarget().getTriple().getVendor() != llvm::Triple::AMD)
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

StringRef SPIRVTargetCodeGenInfo::getLLVMSyncScopeStr(
    const LangOptions &, SyncScope Scope, llvm::AtomicOrdering) const {
  switch (Scope) {
  case SyncScope::HIPSingleThread:
  case SyncScope::SingleScope:
    return "singlethread";
  case SyncScope::HIPWavefront:
  case SyncScope::OpenCLSubGroup:
  case SyncScope::WavefrontScope:
    return "subgroup";
  case SyncScope::HIPCluster:
  case SyncScope::ClusterScope:
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
    IntParams[0] = 1;
  else if (OpenCLName.starts_with("image3d"))
    IntParams[0] = 2;
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

// Gets a spirv.IntegralConstant or spirv.Literal. If IntegralType is present,
// returns an IntegralConstant, otherwise returns a Literal.
static llvm::Type *getInlineSpirvConstant(CodeGenModule &CGM,
                                          llvm::Type *IntegralType,
                                          llvm::APInt Value) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  // Convert the APInt value to an array of uint32_t words
  llvm::SmallVector<uint32_t> Words;

  while (Value.ugt(0)) {
    uint32_t Word = Value.trunc(32).getZExtValue();
    Value.lshrInPlace(32);

    Words.push_back(Word);
  }
  if (Words.size() == 0)
    Words.push_back(0);

  if (IntegralType)
    return llvm::TargetExtType::get(Ctx, "spirv.IntegralConstant",
                                    {IntegralType}, Words);
  return llvm::TargetExtType::get(Ctx, "spirv.Literal", {}, Words);
}

static llvm::Type *getInlineSpirvType(CodeGenModule &CGM,
                                      const HLSLInlineSpirvType *SpirvType) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  llvm::SmallVector<llvm::Type *> Operands;

  for (auto &Operand : SpirvType->getOperands()) {
    using SpirvOperandKind = SpirvOperand::SpirvOperandKind;

    llvm::Type *Result = nullptr;
    switch (Operand.getKind()) {
    case SpirvOperandKind::ConstantId: {
      llvm::Type *IntegralType =
          CGM.getTypes().ConvertType(Operand.getResultType());

      Result = getInlineSpirvConstant(CGM, IntegralType, Operand.getValue());
      break;
    }
    case SpirvOperandKind::Literal: {
      Result = getInlineSpirvConstant(CGM, nullptr, Operand.getValue());
      break;
    }
    case SpirvOperandKind::TypeId: {
      QualType TypeOperand = Operand.getResultType();
      if (const auto *RD = TypeOperand->getAsRecordDecl()) {
        assert(RD->isCompleteDefinition() &&
               "Type completion should have been required in Sema");

        const FieldDecl *HandleField = RD->findFirstNamedDataMember();
        if (HandleField) {
          QualType ResourceType = HandleField->getType();
          if (ResourceType->getAs<HLSLAttributedResourceType>()) {
            TypeOperand = ResourceType;
          }
        }
      }
      Result = CGM.getTypes().ConvertType(TypeOperand);
      break;
    }
    default:
      llvm_unreachable("HLSLInlineSpirvType had invalid operand!");
      break;
    }

    assert(Result);
    Operands.push_back(Result);
  }

  return llvm::TargetExtType::get(Ctx, "spirv.Type", Operands,
                                  {SpirvType->getOpcode(), SpirvType->getSize(),
                                   SpirvType->getAlignment()});
}

llvm::Type *CommonSPIRTargetCodeGenInfo::getHLSLType(
    CodeGenModule &CGM, const Type *Ty,
    const CGHLSLOffsetInfo &OffsetInfo) const {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  if (auto *SpirvType = dyn_cast<HLSLInlineSpirvType>(Ty))
    return getInlineSpirvType(CGM, SpirvType);

  auto *ResType = dyn_cast<HLSLAttributedResourceType>(Ty);
  if (!ResType)
    return nullptr;

  const HLSLAttributedResourceType::Attributes &ResAttrs = ResType->getAttrs();
  switch (ResAttrs.ResourceClass) {
  case llvm::dxil::ResourceClass::UAV:
  case llvm::dxil::ResourceClass::SRV: {
    // TypedBuffer and RawBuffer both need element type
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull())
      return nullptr;

    assert(!ResAttrs.IsROV &&
           "Rasterizer order views not implemented for SPIR-V yet");

    if (!ResAttrs.RawBuffer) {
      // convert element type
      return getSPIRVImageTypeFromHLSLResource(ResAttrs, ContainedTy, CGM);
    }

    if (ResAttrs.IsCounter) {
      llvm::Type *ElemType = llvm::Type::getInt32Ty(Ctx);
      uint32_t StorageClass = /* StorageBuffer storage class */ 12;
      return llvm::TargetExtType::get(Ctx, "spirv.VulkanBuffer", {ElemType},
                                      {StorageClass, true});
    }
    llvm::Type *ElemType = CGM.getTypes().ConvertTypeForMem(ContainedTy);
    llvm::ArrayType *RuntimeArrayType = llvm::ArrayType::get(ElemType, 0);
    uint32_t StorageClass = /* StorageBuffer storage class */ 12;
    bool IsWritable = ResAttrs.ResourceClass == llvm::dxil::ResourceClass::UAV;
    return llvm::TargetExtType::get(Ctx, "spirv.VulkanBuffer",
                                    {RuntimeArrayType},
                                    {StorageClass, IsWritable});
  }
  case llvm::dxil::ResourceClass::CBuffer: {
    QualType ContainedTy = ResType->getContainedType();
    if (ContainedTy.isNull() || !ContainedTy->isStructureType())
      return nullptr;

    llvm::StructType *BufferLayoutTy =
        HLSLBufferLayoutBuilder(CGM).layOutStruct(
            ContainedTy->getAsCanonical<RecordType>(), OffsetInfo);
    uint32_t StorageClass = /* Uniform storage class */ 2;
    return llvm::TargetExtType::get(Ctx, "spirv.VulkanBuffer", {BufferLayoutTy},
                                    {StorageClass, false});
    break;
  }
  case llvm::dxil::ResourceClass::Sampler:
    return llvm::TargetExtType::get(Ctx, "spirv.Sampler");
  }
  return nullptr;
}

static unsigned
getImageFormat(const LangOptions &LangOpts,
               const HLSLAttributedResourceType::Attributes &attributes,
               llvm::Type *SampledType, QualType Ty, unsigned NumChannels) {
  // For images with `Sampled` operand equal to 2, there are restrictions on
  // using the Unknown image format. To avoid these restrictions in common
  // cases, we guess an image format for them based on the sampled type and the
  // number of channels. This is intended to match the behaviour of DXC.
  if (LangOpts.HLSLSpvUseUnknownImageFormat ||
      attributes.ResourceClass != llvm::dxil::ResourceClass::UAV) {
    return 0; // Unknown
  }

  if (SampledType->isIntegerTy(32)) {
    if (Ty->isSignedIntegerType()) {
      if (NumChannels == 1)
        return 24; // R32i
      if (NumChannels == 2)
        return 25; // Rg32i
      if (NumChannels == 4)
        return 21; // Rgba32i
    } else {
      if (NumChannels == 1)
        return 33; // R32ui
      if (NumChannels == 2)
        return 35; // Rg32ui
      if (NumChannels == 4)
        return 30; // Rgba32ui
    }
  } else if (SampledType->isIntegerTy(64)) {
    if (NumChannels == 1) {
      if (Ty->isSignedIntegerType()) {
        return 41; // R64i
      }
      return 40; // R64ui
    }
  } else if (SampledType->isFloatTy()) {
    if (NumChannels == 1)
      return 3; // R32f
    if (NumChannels == 2)
      return 6; // Rg32f
    if (NumChannels == 4)
      return 1; // Rgba32f
  }

  return 0; // Unknown
}

llvm::Type *CommonSPIRTargetCodeGenInfo::getSPIRVImageTypeFromHLSLResource(
    const HLSLAttributedResourceType::Attributes &attributes, QualType Ty,
    CodeGenModule &CGM) const {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  unsigned NumChannels = 1;
  Ty = Ty->getCanonicalTypeUnqualified();
  if (const VectorType *V = dyn_cast<VectorType>(Ty)) {
    NumChannels = V->getNumElements();
    Ty = V->getElementType();
  }
  assert(!Ty->isVectorType() && "We still have a vector type.");

  llvm::Type *SampledType = CGM.getTypes().ConvertTypeForMem(Ty);

  assert((SampledType->isIntegerTy() || SampledType->isFloatingPointTy()) &&
         "The element type for a SPIR-V resource must be a scalar integer or "
         "floating point type.");

  // These parameters correspond to the operands to the OpTypeImage SPIR-V
  // instruction. See
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage.
  SmallVector<unsigned, 6> IntParams(6, 0);

  const char *Name =
      Ty->isSignedIntegerType() ? "spirv.SignedImage" : "spirv.Image";

  // Dim
  switch (attributes.ResourceDimension) {
  case llvm::dxil::ResourceDimension::Dim1D:
    IntParams[0] = 0;
    break;
  case llvm::dxil::ResourceDimension::Dim2D:
    IntParams[0] = 1;
    break;
  case llvm::dxil::ResourceDimension::Dim3D:
    IntParams[0] = 2;
    break;
  case llvm::dxil::ResourceDimension::Cube:
    IntParams[0] = 3;
    break;
  case llvm::dxil::ResourceDimension::Unknown:
    IntParams[0] = 5;
    break;
  }

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
  IntParams[5] = getImageFormat(CGM.getLangOpts(), attributes, SampledType, Ty,
                                NumChannels);

  llvm::TargetExtType *ImageType =
      llvm::TargetExtType::get(Ctx, Name, {SampledType}, IntParams);
  return ImageType;
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createCommonSPIRTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<CommonSPIRTargetCodeGenInfo>(CGM.getTypes());
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createSPIRVTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<SPIRVTargetCodeGenInfo>(CGM.getTypes());
}
