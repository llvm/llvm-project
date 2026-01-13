//===--- AMDGPU.h - AMDGPU ToolChain Implementations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPU_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPU_H

#include "Gnu.h"
#include "clang/Basic/TargetID.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Options/Options.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/TargetParser/TargetParser.h"

#include <map>

namespace clang {
namespace driver {

namespace tools {
namespace amdgpu {

class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("amdgpu::Linker", "ld.lld", TC) {}
  bool isLinkJob() const override { return true; }
  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

void getAMDGPUTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                             const llvm::opt::ArgList &Args,
                             std::vector<StringRef> &Features);

void addFullLTOPartitionOption(const Driver &D, const llvm::opt::ArgList &Args,
                               llvm::opt::ArgStringList &CmdArgs);
} // end namespace amdgpu
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY AMDGPUToolChain : public Generic_ELF {
protected:
  const std::map<options::ID, const StringRef> OptionsDefault;

  Tool *buildLinker() const override;
  StringRef getOptionDefault(options::ID OptID) const {
    auto opt = OptionsDefault.find(OptID);
    assert(opt != OptionsDefault.end() && "No Default for Option");
    return opt->second;
  }

public:
  AMDGPUToolChain(const Driver &D, const llvm::Triple &Triple,
                  const llvm::opt::ArgList &Args);
  unsigned GetDefaultDwarfVersion() const override { return 5; }

  bool IsMathErrnoDefault() const override { return false; }
  bool isCrossCompiling() const override { return true; }
  bool isPICDefault() const override { return true; }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override {
    return false;
  }
  bool isPICDefaultForced() const override { return true; }
  bool SupportsProfiling() const override { return false; }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;

  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args,
                             Action::OffloadKind DeviceOffloadKind) const override;
  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  /// Return whether denormals should be flushed, and treated as 0 by default
  /// for the subtarget.
  static bool getDefaultDenormsAreZeroForTarget(llvm::AMDGPU::GPUKind GPUKind);

  llvm::DenormalMode getDefaultDenormalModeForType(
      const llvm::opt::ArgList &DriverArgs, const JobAction &JA,
      const llvm::fltSemantics *FPType = nullptr) const override;

  static bool isWave64(const llvm::opt::ArgList &DriverArgs,
                       llvm::AMDGPU::GPUKind Kind);
  /// Needed for using lto.
  bool HasNativeLLVMSupport() const override {
    return true;
  }

  /// Needed for translating LTO options.
  const char *getDefaultLinker() const override { return "ld.lld"; }

  /// Should skip sanitize option.
  bool shouldSkipSanitizeOption(const ToolChain &TC,
                                const llvm::opt::ArgList &DriverArgs,
                                StringRef TargetID,
                                const llvm::opt::Arg *A) const;

  /// Uses amdgpu-arch tool to get arch of the system GPU. Will return error
  /// if unable to find one.
  virtual Expected<SmallVector<std::string>>
  getSystemGPUArchs(const llvm::opt::ArgList &Args) const override;

protected:
  /// Check and diagnose invalid target ID specified by -mcpu.
  virtual void checkTargetID(const llvm::opt::ArgList &DriverArgs) const;

  /// The struct type returned by getParsedTargetID.
  struct ParsedTargetIDType {
    std::optional<std::string> OptionalTargetID;
    std::optional<std::string> OptionalGPUArch;
    std::optional<llvm::StringMap<bool>> OptionalFeatures;
  };

  /// Get target ID, GPU arch, and target ID features if the target ID is
  /// specified and valid.
  ParsedTargetIDType
  getParsedTargetID(const llvm::opt::ArgList &DriverArgs) const;

  /// Get GPU arch from -mcpu without checking.
  StringRef getGPUArch(const llvm::opt::ArgList &DriverArgs) const;

  /// Common warning options shared by AMDGPU HIP, OpenCL and OpenMP toolchains.
  /// Language specific warning options should go to derived classes.
  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
};

class LLVM_LIBRARY_VISIBILITY ROCMToolChain : public AMDGPUToolChain {
public:
  ROCMToolChain(const Driver &D, const llvm::Triple &Triple,
                const llvm::opt::ArgList &Args);
  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;

  // Returns a list of device library names shared by different languages
  llvm::SmallVector<BitCodeLibraryInfo, 12>
  getCommonDeviceLibNames(const llvm::opt::ArgList &DriverArgs,
                          const std::string &GPUArch,
                          Action::OffloadKind DeviceOffloadingKind) const;

  SanitizerMask getSupportedSanitizers() const override {
    return SanitizerKind::Address;
  }

  bool diagnoseUnsupportedOption(const llvm::opt::Arg *A,
                                 const llvm::opt::DerivedArgList &DAL,
                                 const llvm::opt::ArgList &DriverArgs,
                                 const char *Value = nullptr) const {
    auto &Diags = getDriver().getDiags();
    bool IsExplicitDevice =
        A->getBaseArg().getOption().matches(options::OPT_Xarch_device);

    if (Value) {
      unsigned DiagID =
          IsExplicitDevice
              ? clang::diag::err_drv_unsupported_option_part_for_target
              : clang::diag::warn_drv_unsupported_option_part_for_target;
      Diags.Report(DiagID) << Value << A->getAsString(DriverArgs)
                           << getTriple().str();
    } else {
      unsigned DiagID =
          IsExplicitDevice
              ? clang::diag::err_drv_unsupported_option_for_target
              : clang::diag::warn_drv_unsupported_option_for_target;
      Diags.Report(DiagID) << A->getAsString(DAL) << getTriple().str();
    }
    return true;
  }

  bool handleSanitizeOption(const ToolChain &TC, llvm::opt::DerivedArgList &DAL,
                            const llvm::opt::ArgList &DriverArgs,
                            StringRef TargetID, const llvm::opt::Arg *A) const {
    if (TargetID.empty())
      return false;
    // If we shouldn't do sanitizing, skip it.
    if (!DriverArgs.hasFlag(options::OPT_fgpu_sanitize,
                            options::OPT_fno_gpu_sanitize, true))
      return true;
    const llvm::opt::Option &Opt = A->getOption();
    // Sanitizer coverage is currently not supported for AMDGPU, so warn/error
    // on every related option.
    if (Opt.matches(options::OPT_fsan_cov_Group)) {
      diagnoseUnsupportedOption(A, DAL, DriverArgs);
    }
    // If this isn't a sanitizer option, don't handle it.
    if (!Opt.matches(options::OPT_fsanitize_EQ))
      return false;

    SmallVector<const char *, 4> SupportedSanitizers;
    SmallVector<const char *, 4> UnSupportedSanitizers;

    for (const char *Value : A->getValues()) {
      SanitizerMask K = parseSanitizerValue(Value, /*Allow Groups*/ false);
      if (K & ROCMToolChain::getSupportedSanitizers())
        SupportedSanitizers.push_back(Value);
      else
        UnSupportedSanitizers.push_back(Value);
    }

    // If there are no supported sanitizers, drop the whole argument.
    if (SupportedSanitizers.empty()) {
      diagnoseUnsupportedOption(A, DAL, DriverArgs);
      return true;
    }
    // If only some sanitizers are unsupported, report each one individually.
    if (!UnSupportedSanitizers.empty()) {
      for (const char *Value : UnSupportedSanitizers) {
        diagnoseUnsupportedOption(A, DAL, DriverArgs, Value);
      }
    }
    // If we know the target arch, check if the sanitizer is supported for it.
    if (shouldSkipSanitizeOption(TC, DriverArgs, TargetID, A))
      return true;

    // Add a new argument with only the supported sanitizers.
    DAL.AddJoinedArg(A, A->getOption(), llvm::join(SupportedSanitizers, ","));
    return true;
  }
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPU_H
