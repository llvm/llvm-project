//===--- SPIRVOpenMP.h - SPIR-V OpenMP Tool Implementations ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_OPENMP_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_OPENMP_H

#include "SPIRV.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace tools {
namespace SPIRVOpenMP {

/// Links device bitcode files with llvm-link and then translates the result
/// to SPIR-V via amd-llvm-spirv (or a versioned/fallback llvm-spirv).
class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
  Linker(const ToolChain &TC)
      : Tool("SPIRVOpenMP::Linker", "spirv-omp-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  /// Builds two chained commands:
  ///   1. llvm-link  — merge all input bitcode files (+ device RTL) into one.
  ///   2. amd-llvm-spirv (or llvm-spirv) — translate the merged BC to SPIR-V.
  void constructLinkAndEmitSpirvCommand(
      Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
      const InputInfo &Output, const llvm::opt::ArgList &Args) const;

  /// Returns the path to the best available SPIR-V translator binary, trying
  /// versioned AMD-specific names before falling back to the generic one.
  std::string findSpirvTranslator(const llvm::opt::ArgList &Args) const;
};

} // namespace SPIRVOpenMP
} // namespace tools

namespace toolchains {

/// Toolchain for compiling OpenMP device code to AMDGCN-flavored SPIR-V.
///
/// Inherits basic SPIR-V toolchain infrastructure from SPIRVToolChain and
/// delegates host-specific queries (include paths, warnings, C++ stdlib, etc.)
/// to the accompanying host toolchain.
class LLVM_LIBRARY_VISIBILITY SPIRVOpenMPToolChain : public SPIRVToolChain {
public:
  SPIRVOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                       const ToolChain &HostTC,
                       const llvm::opt::ArgList &Args);

  const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  void addClangTargetOptions(
      const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
      BoundArch BA, Action::OffloadKind DeviceOffloadingKind) const override;

  void addClangWarningOptions(
      llvm::opt::ArgStringList &CC1Args) const override;

  CXXStdlibType GetCXXStdlibType(
      const llvm::opt::ArgList &Args) const override;

  void AddClangSystemIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CC1Args) const override;

  llvm::SmallVector<BitCodeLibraryInfo, 12>
  getDeviceLibs(const llvm::opt::ArgList &Args,
                Action::OffloadKind DeviceOffloadKind) const override;

  SanitizerMask getSupportedSanitizers() const override;

  VersionTuple computeMSVCVersion(const Driver *D,
                                  const llvm::opt::ArgList &Args) const override;

  void adjustDebugInfoKind(
      llvm::codegenoptions::DebugInfoKind &DebugInfoKind,
      const llvm::opt::ArgList &Args) const override;

  const ToolChain &HostTC;

protected:
  Tool *buildLinker() const override;
};

} // namespace toolchains
} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_OPENMP_H
