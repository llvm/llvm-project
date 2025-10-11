//===--- Solaris.h - Solaris ToolChain Implementations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SOLARIS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SOLARIS_H

#include "Gnu.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace tools {

/// Directly call Solaris assembler and linker
namespace solaris {
class LLVM_LIBRARY_VISIBILITY Assembler final : public gnutools::Assembler {
public:
  Assembler(const ToolChain &TC) : gnutools::Assembler(TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

bool isLinkerSolarisLinkEditor(const ToolChain &TC,
                               const llvm::opt::ArgList &Args);

class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("solaris::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  std::string getLinkerPath(const llvm::opt::ArgList &Args) const;

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

enum class LinkerExpectations {
  GnuLdCompatibleArgParser,
  /// The formal name for the built-in ld on Solaris.
  SolarisLinkEditor,
};

/// We use Solaris's built-in linker by default. It has a unique command line
/// syntax and specific limitations. By contrast, other linkers such as lld,
/// Mold, and Wild are compatible with GNU ld's command line syntax. Knowing
/// _which_ linker to use is sufficient to determine the expectations of that
/// linker. Rather than spread ad-hoc string comparisons all over the driver, we
/// encapsulate the details of differences in the chosen linker here.
class LinkerDetermination final {
  LinkerDetermination(std::string Linker, LinkerExpectations Expectations)
      : Linker(Linker), Expectations(Expectations) {}

public:
  std::string Linker;
  LinkerExpectations Expectations;

  /// Choose the correct linker based on arguments and compile-time options
  /// recorded in the ToolChain.
  static LinkerDetermination make(const ToolChain &TC,
                                  const llvm::opt::ArgList &Args,
                                  bool EmitDiagnostics);
};

} // end namespace solaris
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY Solaris : public Generic_ELF {
public:
  Solaris(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  void
  addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;

  SanitizerMask getSupportedSanitizers() const override;

  const char *getDefaultLinker() const override;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SOLARIS_H
