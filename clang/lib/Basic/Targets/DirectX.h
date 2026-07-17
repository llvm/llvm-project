//===--- DirectX.h - Declare DirectX target feature support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares DXIL TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_DIRECTX_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_DIRECTX_H
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

static const LangASMap DirectXAddrSpaceMap = {
    {LangAS::Default, 0},
    {LangAS::opencl_global, 1},
    {LangAS::opencl_local, 3},
    {LangAS::opencl_constant, 2},
    {LangAS::opencl_private, 0},
    {LangAS::opencl_generic, 4},
    {LangAS::opencl_global_device, 5},
    {LangAS::opencl_global_host, 6},
    {LangAS::cuda_device, 0},
    {LangAS::cuda_constant, 0},
    {LangAS::cuda_shared, 0},
    // SYCL address space values for this map are dummy
    {LangAS::sycl_global, 0},
    {LangAS::sycl_global_device, 0},
    {LangAS::sycl_global_host, 0},
    {LangAS::sycl_local, 0},
    {LangAS::sycl_private, 0},
    {LangAS::ptr32_sptr, 0},
    {LangAS::ptr32_uptr, 0},
    {LangAS::ptr64, 0},
    {LangAS::hlsl_groupshared, 3},
    {LangAS::hlsl_constant, 2},
    {LangAS::hlsl_private, 0},
    {LangAS::hlsl_device, 0},
    {LangAS::hlsl_input, 0},
    {LangAS::hlsl_output, 0},
    {LangAS::hlsl_push_constant, 0},
    {LangAS::wasm_funcref, 0},
};

class LLVM_LIBRARY_VISIBILITY DirectXTargetInfo : public TargetInfo {
public:
  DirectXTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    TLSSupported = false;
    VLASupported = false;
    AddrSpaceMap = &DirectXAddrSpaceMap;
    UseAddrSpaceMapMangling = true;
    HasFastHalfType = true;
    HasFloat16 = true;
    NoAsmVariants = true;
    VectorsAreElementAligned = true;
    PlatformMinVersion = Triple.getOSVersion();
    PlatformName = llvm::Triple::getOSTypeName(Triple.getOS());
    resetDataLayout();
    TheCXXABI.set(TargetCXXABI::GenericItanium);
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool hasFeature(StringRef Feature) const override {
    return Feature == "directx";
  }

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override;

  std::string_view getClobbers() const override { return ""; }

  ArrayRef<const char *> getGCCRegNames() const override { return {}; }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return true;
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return {};
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  void adjust(DiagnosticsEngine &Diags, LangOptions &Opts,
              const TargetInfo *Aux) override {
    TargetInfo::adjust(Diags, Opts, Aux);
    // The static values this addresses do not apply outside of the same thread
    // This protection is neither available nor needed
    Opts.ThreadsafeStatics = false;
  }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_DIRECTX_H
