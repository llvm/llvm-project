//===--- Next32.h - Declare Next32 target feature support ---------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares Next32 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_NEXT32_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_NEXT32_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

static const unsigned Next32AddrSpaceMap[] = {
    0, // Default
    0, // opencl_global
    0, // opencl_local
    0, // opencl_constant
    0, // opencl_private
    0, // opencl_generic
    0, // opencl_global_device
    0, // opencl_global_host
    0, // cuda_device
    0, // cuda_constant
    0, // cuda_shared
    0, // sycl_global
    0, // sycl_global_device
    0, // sycl_global_host
    0, // sycl_local
    0, // sycl_private
    0, // ptr32_sptr
    0, // ptr32_uptr
    0, // ptr64
    0, // hlsl_groupshared
    // Wasm address space values for this target are dummy values,
    // as it is only enabled for Wasm targets.
    20,  // wasm_funcref
    273, // next32_tls
    274, // next32_global
    275, // next32_constant
    3,   // next32_local
};

class LLVM_LIBRARY_VISIBILITY Next32TargetInfo : public TargetInfo {
  static const Builtin::Info BuiltinInfo[];

public:
  Next32TargetInfo(const llvm::Triple &Triple, const TargetOptions &);

  bool isValidCPUName(StringRef Name) const override;

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;

  bool setCPU(const std::string &Name) override { return isValidCPUName(Name); }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  ArrayRef<Builtin::Info> getTargetBuiltins() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  std::string_view getClobbers() const override { return ""; }

  bool hasInt128Type() const override { return true; }

  bool hasBitIntType() const override { return true; }

  // This could be any amount, 1024 is arbitrary and should suffice
  size_t getMaxBitIntWidth() const override { return 1024; }

  ArrayRef<const char *> getGCCRegNames() const override {
    static const char *const GCCRegNames[] = {"r1"};
    return llvm::ArrayRef(GCCRegNames);
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return std::nullopt;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    return false;
  }

  bool isValidFeatureName(StringRef Name) const override;

  bool hasFeature(StringRef Feature) const override;

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;
};
} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_NEXT32_H
