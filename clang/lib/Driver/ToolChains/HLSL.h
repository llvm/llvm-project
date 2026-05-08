//===--- HLSL.h - HLSL ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HLSL_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HLSL_H

#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

namespace tools {

namespace hlsl {
class LLVM_LIBRARY_VISIBILITY Validator : public Tool {
public:
  Validator(const ToolChain &TC)
      : Tool("hlsl::Validator", TC.getTriple().isSPIRV() ? "spirv-val" : "dxv",
             TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY MetalConverter : public Tool {
public:
  MetalConverter(const ToolChain &TC)
      : Tool("hlsl::MetalConverter", "metal-shaderconverter", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY LLVMObjcopy : public Tool {
public:
  LLVMObjcopy(const ToolChain &TC)
      : Tool("hlsl::LLVMObjcopy", "llvm-objcopy", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // namespace hlsl
} // namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY HLSLToolChain : public ToolChain {
public:
  HLSLToolChain(const Driver &D, const llvm::Triple &Triple,
                const llvm::opt::ArgList &Args);
  Tool *getTool(Action::ActionClass AC) const override;

  bool isPICDefault() const override { return false; }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override {
    return false;
  }
  bool isPICDefaultForced() const override { return false; }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  static std::optional<std::string> parseTargetProfile(StringRef TargetProfile);

  struct ValidationInfo {
    bool NeedsValidation = false;
    bool ProducesOutput = false;
  };

  /// Returns information about whether validation is required and whether the
  /// validator produces output. When Diagnose is true, emits a warning if the
  /// required validator executable cannot be found.
  ValidationInfo getValidationInfo(llvm::opt::DerivedArgList &Args,
                                   bool Diagnose = true) const;
  bool requiresBinaryTranslation(llvm::opt::DerivedArgList &Args) const;
  bool requiresObjcopy(llvm::opt::DerivedArgList &Args) const;

  /// Determines whether the given action class is the last job that produces
  /// an output file. This is used to decide whether to write to the -Fo
  /// output path or to a temporary file.
  ///
  /// For example, spirv-val is a pure validator that runs after the compile
  /// step but doesn't produce output, so the compile step is the last
  /// output-producing job. For DXIL, dxv validates and signs, producing the
  /// final output.
  bool isLastOutputProducingJob(llvm::opt::DerivedArgList &Args,
                                Action::ActionClass AC) const;

  // Set default DWARF version to 4 for DXIL uses version 4.
  unsigned GetDefaultDwarfVersion() const override { return 4; }

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;

private:
  mutable std::unique_ptr<tools::hlsl::Validator> Validator;
  mutable std::unique_ptr<tools::hlsl::MetalConverter> MetalConverter;
  mutable std::unique_ptr<tools::hlsl::LLVMObjcopy> LLVMObjcopy;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_HLSL_H
