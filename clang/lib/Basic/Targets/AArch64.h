//===--- AArch64.h - Declare AArch64 target feature support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares AArch64 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_AARCH64_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_AARCH64_H

#include "OSTargets.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include <optional>

namespace clang {
namespace targets {

enum AArch64AddrSpace { ptr32_sptr = 270, ptr32_uptr = 271, ptr64 = 272 };

static const unsigned ARM64AddrSpaceMap[] = {
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
    static_cast<unsigned>(AArch64AddrSpace::ptr32_sptr),
    static_cast<unsigned>(AArch64AddrSpace::ptr32_uptr),
    static_cast<unsigned>(AArch64AddrSpace::ptr64),
    0, // hlsl_groupshared
    0, // hlsl_constant
    0, // hlsl_private
    0, // hlsl_device
    0, // hlsl_input
    0, // hlsl_output
    0, // hlsl_push_constant
    // Wasm address space values for this target are dummy values,
    // as it is only enabled for Wasm targets.
    20, // wasm_funcref
};

using AArch64FeatureSet = llvm::SmallDenseSet<StringRef, 32>;

class LLVM_LIBRARY_VISIBILITY AArch64TargetInfo : public TargetInfo {
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char *const GCCRegNames[];

  enum FPUModeEnum {
    FPUMode = (1 << 0),
    NeonMode = (1 << 1),
    SveMode = (1 << 2),
  };

  unsigned FPU = FPUMode;
  llvm::AArch64::ExtensionBitset EnabledExtensions;
  unsigned ExplicitFPUCompat = 0;
  bool HasNoFP = false;
  bool HasNoNeon = false;
  bool HasNoSVE = false;
  bool HasFMV = true;

  const llvm::AArch64::ArchInfo *ArchInfo = &llvm::AArch64::ARMV8A;

  AArch64FeatureSet HasFeatureLookup;

  void computeFeatureLookup();
  bool hasExtension(llvm::AArch64::ArchExtKind Ext) const;
  bool hasPAuth() const;
  bool hasFullFP16() const;
  void setFeatureStateFromEnabledExtensions(
      const llvm::AArch64::ExtensionBitset &EnabledExtensions);

protected:
  std::string ABI;

public:
  AArch64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  StringRef getABI() const override;
  bool setABI(const std::string &Name) override;
  bool hasFeatureEnabled(const llvm::StringMap<bool> &Features,
                         StringRef Name) const override;

  bool validateBranchProtection(StringRef Spec, StringRef Arch,
                                BranchProtectionInfo &BPI,
                                const LangOptions &LO,
                                StringRef &Err) const override;

  bool isValidCPUName(StringRef Name) const override;
  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;
  bool setCPU(const std::string &Name) override;

  llvm::APInt getFMVPriority(ArrayRef<StringRef> Features) const override;

  bool useFP16ConversionIntrinsics() const override {
    return false;
  }

  void getTargetDefinesARMV81A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV82A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV83A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV84A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV85A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV86A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV87A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV88A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV89A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV9A(const LangOptions &Opts,
                              MacroBuilder &Builder) const;
  void getTargetDefinesARMV91A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV92A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV93A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV94A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV95A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV96A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefinesARMV97A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override;

  std::optional<std::pair<unsigned, unsigned>>
  getVScaleRange(const LangOptions &LangOpts, ArmStreamingKind Mode,
                 llvm::StringMap<bool> *FeatureMap = nullptr) const override;
  bool doesFeatureAffectCodeGen(StringRef Name) const override;
  bool validateCpuSupports(StringRef FeatureStr) const override;
  bool hasFeature(StringRef Feature) const override;
  void setFeatureEnabled(llvm::StringMap<bool> &Features, StringRef Name,
                         bool Enabled) const override;
  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;
  ParsedTargetAttr parseTargetAttr(StringRef Str) const override;
  bool supportsTargetAttributeTune() const override { return true; }
  bool supportsCpuSupports() const override { return true; }
  bool checkArithmeticFenceSupported() const override { return true; }

  bool hasBFloat16Type() const override;

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override;

  bool isCLZForZeroUndef() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override;

  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  std::string convertConstraint(const char *&Constraint) const override;

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override;
  bool
  validateConstraintModifier(StringRef Constraint, char Modifier, unsigned Size,
                             std::string &SuggestedModifier) const override;
  std::string_view getClobbers() const override;

  StringRef getConstraintRegister(StringRef Constraint,
                                  StringRef Expression) const override {
    return Expression;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override;

  bool validatePointerAuthKey(const llvm::APSInt &value) const override;

  const char *getBFloat16Mangling() const override { return "u6__bf16"; };

  std::pair<unsigned, unsigned> hardwareInterferenceSizes() const override {
    return std::make_pair(256, 64);
  }

  bool hasInt128Type() const override;

  bool hasBitIntType() const override { return true; }

  bool validateTarget(DiagnosticsEngine &Diags) const override;

  bool validateGlobalRegisterVariable(StringRef RegName, unsigned RegSize,
                                      bool &HasSizeMismatch) const override;

  uint64_t getPointerWidthV(LangAS AddrSpace) const override {
    if (AddrSpace == LangAS::ptr32_sptr || AddrSpace == LangAS::ptr32_uptr)
      return 32;
    if (AddrSpace == LangAS::ptr64)
      return 64;
    return PointerWidth;
  }

  uint64_t getPointerAlignV(LangAS AddrSpace) const override {
    return getPointerWidthV(AddrSpace);
  }
};

class LLVM_LIBRARY_VISIBILITY AArch64leTargetInfo : public AArch64TargetInfo {
public:
  AArch64leTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

template <>
inline bool
LinuxTargetInfo<AArch64leTargetInfo>::setABI(const std::string &Name) {
  if (Name == "pauthtest") {
    ABI = Name;
    return true;
  }
  return AArch64leTargetInfo::setABI(Name);
}

class LLVM_LIBRARY_VISIBILITY WindowsARM64TargetInfo
    : public WindowsTargetInfo<AArch64leTargetInfo> {
  const llvm::Triple Triple;

public:
  WindowsARM64TargetInfo(const llvm::Triple &Triple,
                         const TargetOptions &Opts);

  BuiltinVaListKind getBuiltinVaListKind() const override;

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override;
};

// Windows ARM, MS (C++) ABI
class LLVM_LIBRARY_VISIBILITY MicrosoftARM64TargetInfo
    : public WindowsARM64TargetInfo {
public:
  MicrosoftARM64TargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts);

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
  TargetInfo::CallingConvKind
  getCallingConvKind(bool ClangABICompat4) const override;

  unsigned getMinGlobalAlign(uint64_t TypeSize,
                             bool HasNonWeakDef) const override;
};

// ARM64 MinGW target
class LLVM_LIBRARY_VISIBILITY MinGWARM64TargetInfo
    : public WindowsARM64TargetInfo {
public:
  MinGWARM64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);
};

class LLVM_LIBRARY_VISIBILITY AArch64beTargetInfo : public AArch64TargetInfo {
public:
  AArch64beTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

void getAppleMachOAArch64Defines(MacroBuilder &Builder, const LangOptions &Opts,
                                 const llvm::Triple &Triple);

class LLVM_LIBRARY_VISIBILITY AppleMachOAArch64TargetInfo
    : public AppleMachOTargetInfo<AArch64leTargetInfo> {
public:
  AppleMachOAArch64TargetInfo(const llvm::Triple &Triple,
                              const TargetOptions &Opts);

protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override;
};

class LLVM_LIBRARY_VISIBILITY DarwinAArch64TargetInfo
    : public DarwinTargetInfo<AArch64leTargetInfo> {
public:
  DarwinAArch64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  BuiltinVaListKind getBuiltinVaListKind() const override;

 protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override;
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_AARCH64_H
