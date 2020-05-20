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
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/TargetParser.h"

#include <map>

namespace clang {
namespace driver {

/// A class to find a viable ROCM installation
/// TODO: Generalize to handle libclc.
class RocmInstallationDetector {
private:
  struct ConditionalLibrary {
    SmallString<0> On;
    SmallString<0> Off;

    bool isValid() const {
      return !On.empty() && !Off.empty();
    }

    StringRef get(bool Enabled) const {
      assert(isValid());
      return Enabled ? On : Off;
    }
  };

  const Driver &D;
  bool IsValid = false;
  //RocmVersion Version = RocmVersion::UNKNOWN;
  SmallString<0> InstallPath;
  //SmallString<0> BinPath;
  SmallString<0> LibPath;
  SmallString<0> LibDevicePath;
  SmallString<0> IncludePath;
  llvm::StringMap<std::string> LibDeviceMap;

  // Libraries that are always linked.
  SmallString<0> OCML;
  SmallString<0> OCKL;

  // Libraries that are always linked depending on the language
  SmallString<0> OpenCL;
  SmallString<0> HIP;

  // Libraries swapped based on compile flags.
  ConditionalLibrary WavefrontSize64;
  ConditionalLibrary FiniteOnly;
  ConditionalLibrary UnsafeMath;
  ConditionalLibrary DenormalsAreZero;
  ConditionalLibrary CorrectlyRoundedSqrt;

  bool allGenericLibsValid() const {
    return !OCML.empty() && !OCKL.empty() && !OpenCL.empty() && !HIP.empty() &&
           WavefrontSize64.isValid() && FiniteOnly.isValid() &&
           UnsafeMath.isValid() && DenormalsAreZero.isValid() &&
           CorrectlyRoundedSqrt.isValid();
  }

  // CUDA architectures for which we have raised an error in
  // CheckRocmVersionSupportsArch.
  mutable llvm::SmallSet<CudaArch, 4> ArchsWithBadVersion;

  void scanLibDevicePath();

public:
  RocmInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                           const llvm::opt::ArgList &Args);

  /// Add arguments needed to link default bitcode libraries.
  void addCommonBitcodeLibCC1Args(const llvm::opt::ArgList &DriverArgs,
                                  llvm::opt::ArgStringList &CC1Args,
                                  StringRef LibDeviceFile, bool Wave64,
                                  bool DAZ, bool FiniteOnly, bool UnsafeMathOpt,
                                  bool FastRelaxedMath, bool CorrectSqrt) const;

  /// Emit an error if Version does not support the given Arch.
  ///
  /// If either Version or Arch is unknown, does not emit an error.  Emits at
  /// most one error per Arch.
  void CheckRocmVersionSupportsArch(CudaArch Arch) const;

  /// Check whether we detected a valid Rocm install.
  bool isValid() const { return IsValid; }
  /// Print information about the detected CUDA installation.
  void print(raw_ostream &OS) const;

  /// Get the detected Rocm install's version.
  // RocmVersion version() const { return Version; }

  /// Get the detected Rocm installation path.
  StringRef getInstallPath() const { return InstallPath; }

  /// Get the detected path to Rocm's bin directory.
  // StringRef getBinPath() const { return BinPath; }

  /// Get the detected Rocm Include path.
  StringRef getIncludePath() const { return IncludePath; }

  /// Get the detected Rocm library path.
  StringRef getLibPath() const { return LibPath; }

  /// Get the detected Rocm device library path.
  StringRef getLibDevicePath() const { return LibDevicePath; }

  StringRef getOCMLPath() const {
    assert(!OCML.empty());
    return OCML;
  }

  StringRef getOCKLPath() const {
    assert(!OCKL.empty());
    return OCKL;
  }

  StringRef getOpenCLPath() const {
    assert(!OpenCL.empty());
    return OpenCL;
  }

  StringRef getHIPPath() const {
    assert(!HIP.empty());
    return HIP;
  }

  StringRef getWavefrontSize64Path(bool Enabled) const {
    return WavefrontSize64.get(Enabled);
  }

  StringRef getFiniteOnlyPath(bool Enabled) const {
    return FiniteOnly.get(Enabled);
  }

  StringRef getUnsafeMathPath(bool Enabled) const {
    return UnsafeMath.get(Enabled);
  }

  StringRef getDenormalsAreZeroPath(bool Enabled) const {
    return DenormalsAreZero.get(Enabled);
  }

  StringRef getCorrectlyRoundedSqrtPath(bool Enabled) const {
    return CorrectlyRoundedSqrt.get(Enabled);
  }

  /// Get libdevice file for given architecture
  std::string getLibDeviceFile(StringRef Gpu) const {
    return LibDeviceMap.lookup(Gpu);
  }
};

namespace tools {
namespace amdgpu {

class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC) : GnuTool("amdgpu::Linker", "ld.lld", TC) {}
  bool isLinkJob() const override { return true; }
  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

void getAMDGPUTargetFeatures(const Driver &D, const llvm::opt::ArgList &Args,
                             std::vector<StringRef> &Features);

} // end namespace amdgpu
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY AMDGPUToolChain : public Generic_ELF {
protected:
  const std::map<options::ID, const StringRef> OptionsDefault;

  Tool *buildLinker() const override;
  const StringRef getOptionDefault(options::ID OptID) const {
    auto opt = OptionsDefault.find(OptID);
    assert(opt != OptionsDefault.end() && "No Default for Option");
    return opt->second;
  }

public:
  AMDGPUToolChain(const Driver &D, const llvm::Triple &Triple,
                  const llvm::opt::ArgList &Args);
  unsigned GetDefaultDwarfVersion() const override { return 4; }
  bool IsIntegratedAssemblerDefault() const override { return true; }
  bool IsMathErrnoDefault() const override { return false; }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;

  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args,
                             Action::OffloadKind DeviceOffloadKind) const override;

  /// Return whether denormals should be flushed, and treated as 0 by default
  /// for the subtarget.
  static bool getDefaultDenormsAreZeroForTarget(llvm::AMDGPU::GPUKind GPUKind);

  llvm::DenormalMode getDefaultDenormalModeForType(
      const llvm::opt::ArgList &DriverArgs, const JobAction &JA,
      const llvm::fltSemantics *FPType = nullptr) const override;

  static bool isWave64(const llvm::opt::ArgList &DriverArgs,
                       llvm::AMDGPU::GPUKind Kind);
};

class LLVM_LIBRARY_VISIBILITY ROCMToolChain : public AMDGPUToolChain {
protected:
  RocmInstallationDetector RocmInstallation;

public:
  ROCMToolChain(const Driver &D, const llvm::Triple &Triple,
                const llvm::opt::ArgList &Args);
  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDGPU_H
