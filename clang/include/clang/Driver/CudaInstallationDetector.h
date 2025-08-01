//===-- CudaInstallationDetector.h - Cuda Instalation Detector --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_CUDAINSTALLATIONDETECTOR_H
#define LLVM_CLANG_DRIVER_CUDAINSTALLATIONDETECTOR_H

#include "clang/Basic/Cuda.h"
#include "clang/Driver/Driver.h"
#include <bitset>

namespace clang {
namespace driver {

/// A class to find a viable CUDA installation
class CudaInstallationDetector {
private:
  const Driver &D;
  bool IsValid = false;
  CudaVersion Version = CudaVersion::UNKNOWN;
  std::string InstallPath;
  std::string BinPath;
  std::string LibDevicePath;
  std::string IncludePath;
  llvm::StringMap<std::string> LibDeviceMap;

  // CUDA architectures for which we have raised an error in
  // CheckCudaVersionSupportsArch.
  mutable std::bitset<(int)OffloadArch::LAST> ArchsWithBadVersion;

public:
  CudaInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                           const llvm::opt::ArgList &Args);

  void AddCudaIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const;

  /// Emit an error if Version does not support the given Arch.
  ///
  /// If either Version or Arch is unknown, does not emit an error.  Emits at
  /// most one error per Arch.
  void CheckCudaVersionSupportsArch(OffloadArch Arch) const;

  /// Check whether we detected a valid Cuda install.
  bool isValid() const { return IsValid; }
  /// Print information about the detected CUDA installation.
  void print(raw_ostream &OS) const;

  /// Get the detected Cuda install's version.
  CudaVersion version() const {
    return Version == CudaVersion::NEW ? CudaVersion::PARTIALLY_SUPPORTED
                                       : Version;
  }
  /// Get the detected Cuda installation path.
  StringRef getInstallPath() const { return InstallPath; }
  /// Get the detected path to Cuda's bin directory.
  StringRef getBinPath() const { return BinPath; }
  /// Get the detected Cuda Include path.
  StringRef getIncludePath() const { return IncludePath; }
  /// Get the detected Cuda device library path.
  StringRef getLibDevicePath() const { return LibDevicePath; }
  /// Get libdevice file for given architecture
  std::string getLibDeviceFile(StringRef Gpu) const {
    return LibDeviceMap.lookup(Gpu);
  }
  void WarnIfUnsupportedVersion() const;
};

} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_DRIVER_CUDAINSTALLATIONDETECTOR_H
