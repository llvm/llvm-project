//===--- TargetUtils.cpp - Shared target-specific AST queries ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/TargetUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/TargetParser/Triple.h"

namespace clang::CodeGenUtils {

//===----------------------------------------------------------------------===//
// Library builtins (PowerPC64 and AIX)
//===----------------------------------------------------------------------===//

StringRef getTargetBuiltinLibFunctionName(unsigned BuiltinID,
                                          const TargetInfo &Target) {
  const llvm::Triple &Triple = Target.getTriple();

  // TODO: This mutation should also be applied to other targets other than
  // PPC, after backend supports IEEE 128-bit style libcalls.
  if (Triple.isPPC64() &&
      &Target.getLongDoubleFormat() == &llvm::APFloat::IEEEquad()) {
    // TODO: This list should be expanded or refactored after all
    // GCC-compatible std libcall builtins are implemented.
    static const llvm::SmallDenseMap<unsigned, StringRef, 64> F128Builtins{
        {Builtin::BI__builtin___fprintf_chk, "__fprintf_chkieee128"},
        {Builtin::BI__builtin___printf_chk, "__printf_chkieee128"},
        {Builtin::BI__builtin___snprintf_chk, "__snprintf_chkieee128"},
        {Builtin::BI__builtin___sprintf_chk, "__sprintf_chkieee128"},
        {Builtin::BI__builtin___vfprintf_chk, "__vfprintf_chkieee128"},
        {Builtin::BI__builtin___vprintf_chk, "__vprintf_chkieee128"},
        {Builtin::BI__builtin___vsnprintf_chk, "__vsnprintf_chkieee128"},
        {Builtin::BI__builtin___vsprintf_chk, "__vsprintf_chkieee128"},
        {Builtin::BI__builtin_fprintf, "__fprintfieee128"},
        {Builtin::BI__builtin_printf, "__printfieee128"},
        {Builtin::BI__builtin_snprintf, "__snprintfieee128"},
        {Builtin::BI__builtin_sprintf, "__sprintfieee128"},
        {Builtin::BI__builtin_vfprintf, "__vfprintfieee128"},
        {Builtin::BI__builtin_vprintf, "__vprintfieee128"},
        {Builtin::BI__builtin_vsnprintf, "__vsnprintfieee128"},
        {Builtin::BI__builtin_vsprintf, "__vsprintfieee128"},
        {Builtin::BI__builtin_fscanf, "__fscanfieee128"},
        {Builtin::BI__builtin_scanf, "__scanfieee128"},
        {Builtin::BI__builtin_sscanf, "__sscanfieee128"},
        {Builtin::BI__builtin_vfscanf, "__vfscanfieee128"},
        {Builtin::BI__builtin_vscanf, "__vscanfieee128"},
        {Builtin::BI__builtin_vsscanf, "__vsscanfieee128"},
        {Builtin::BI__builtin_nexttowardf128, "__nexttowardieee128"},
    };
    return F128Builtins.lookup(BuiltinID);
  }

  // The AIX library functions frexpl, ldexpl, and modfl are for 128-bit
  // IBM 'long double' (i.e. __ibm128). Map to the 'double' versions
  // if it is 64-bit 'long double' mode.
  if (Triple.isOSAIX() &&
      &Target.getLongDoubleFormat() == &llvm::APFloat::IEEEdouble()) {
    static const llvm::SmallDenseMap<unsigned, StringRef, 4>
        AIXLongDouble64Builtins{
            {Builtin::BI__builtin_frexpl, "frexp"},
            {Builtin::BI__builtin_ldexpl, "ldexp"},
            {Builtin::BI__builtin_modfl, "modf"},
    };
    return AIXLongDouble64Builtins.lookup(BuiltinID);
  }

  return StringRef();
}

//===----------------------------------------------------------------------===//
// Arm and AArch64
//===----------------------------------------------------------------------===//

static bool isStreamingCompatible(const FunctionDecl *F) {
  if (const auto *T = F->getType()->getAs<FunctionProtoType>())
    return T->getAArch64SMEAttributes() &
           FunctionType::SME_PStateSMCompatibleMask;
  return false;
}

ArmSMEInlinability getArmSMEInlinability(const FunctionDecl *Caller,
                                         const FunctionDecl *Callee) {
  bool CallerIsStreaming =
      IsArmStreamingFunction(Caller, /*IncludeLocallyStreaming=*/true);
  bool CalleeIsStreaming =
      IsArmStreamingFunction(Callee, /*IncludeLocallyStreaming=*/true);
  bool CallerIsStreamingCompatible = isStreamingCompatible(Caller);
  bool CalleeIsStreamingCompatible = isStreamingCompatible(Callee);

  ArmSMEInlinability Inlinability = ArmSMEInlinability::Ok;

  if (!CalleeIsStreamingCompatible &&
      (CallerIsStreaming != CalleeIsStreaming || CallerIsStreamingCompatible)) {
    if (CalleeIsStreaming)
      Inlinability |= ArmSMEInlinability::ErrorIncompatibleStreamingModes;
    else
      Inlinability |= ArmSMEInlinability::WarnIncompatibleStreamingModes;
  }
  if (auto *NewAttr = Callee->getAttr<ArmNewAttr>()) {
    if (NewAttr->isNewZA())
      Inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZA;
    if (NewAttr->isNewZT0())
      Inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZT0;
  }

  return Inlinability;
}

bool hasExtraNeonArgument(unsigned BuiltinID) {
  // Required by the headers included below, but not in this particular
  // function.
  [[maybe_unused]] int PtrArgNum = -1;
  [[maybe_unused]] bool HasConstPtr = false;

  // The mask encodes the type. We don't care about the actual value. Instead,
  // we just check whether its been set.
  uint64_t mask = 0;
  switch (BuiltinID) {
#define GET_NEON_OVERLOAD_CHECK
#include "clang/Basic/arm_fp16.inc"
#include "clang/Basic/arm_neon.inc"
#undef GET_NEON_OVERLOAD_CHECK
  // Non-neon builtins for controling VFP that take extra argument for
  // discriminating the type.
  case ARM::BI__builtin_arm_vcvtr_f:
  case ARM::BI__builtin_arm_vcvtr_d:
    mask = 1;
  }

  return mask != 0;
}

bool isAAPCS(const TargetInfo &TargetInfo) {
  return TargetInfo.getABI().starts_with("aapcs");
}

//===----------------------------------------------------------------------===//
// X86
//===----------------------------------------------------------------------===//

// Many of MSVC builtins are on x64, ARM and AArch64; to avoid repeating code,
// we handle them here.
std::optional<MSVCIntrin>
translateX86ToMsvcIntrin(unsigned BuiltinID) {
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::X86::BI_BitScanForward:
  case clang::X86::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::X86::BI_BitScanReverse:
  case clang::X86::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::X86::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::X86::BI_InterlockedCompareExchange128:
    return MSVCIntrin::_InterlockedCompareExchange128;
  case clang::X86::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::X86::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::X86::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::X86::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::X86::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::X86::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::X86::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  }
  llvm_unreachable("must return from switch");
}

//===----------------------------------------------------------------------===//
// AMDGPU
//===----------------------------------------------------------------------===//

bool requiresAMDGPUProtectedVisibility(const Decl *D,
                                       bool HasHiddenVisibility) {
  if (!HasHiddenVisibility)
    return false;

  return !D->hasAttr<OMPDeclareTargetDeclAttr>() &&
         (D->hasAttr<DeviceKernelAttr>() ||
          (isa<FunctionDecl>(D) && D->hasAttr<CUDAGlobalAttr>()) ||
          (isa<VarDecl>(D) &&
           (D->hasAttr<CUDADeviceAttr>() || D->hasAttr<CUDAConstantAttr>() ||
            cast<VarDecl>(D)->getType()->isCUDADeviceBuiltinSurfaceType() ||
            cast<VarDecl>(D)->getType()->isCUDADeviceBuiltinTextureType())));
}

//===----------------------------------------------------------------------===//
// SPIR-V
//===----------------------------------------------------------------------===//

bool spirNullPointerNeedsGenericCast(QualType QT, const llvm::Triple &Triple) {
  // LLVM address space of the SPIR-V CodeSectionINTEL storage class.
  constexpr unsigned SPIRVCodeSectionINTELAddrSpace = 9;
  LangAS AS = QT->getUnqualifiedDesugaredType()->isNullPtrType()
                  ? LangAS::Default
                  : QT->getPointeeType().getAddressSpace();
  if (AS == LangAS::Default || AS == LangAS::opencl_generic ||
      AS == LangAS::opencl_constant)
    return false;
  // As per SPV_INTEL_function_pointers, it is illegal to addrspacecast
  // function pointers to/from the generic AS.
  return !(Triple.isSPIRV() && isTargetAddressSpace(AS) &&
           toTargetAddressSpace(AS) == SPIRVCodeSectionINTELAddrSpace);
}

} // namespace clang::CodeGenUtils
