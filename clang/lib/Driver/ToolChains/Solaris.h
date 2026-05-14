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
  Assembler(const ToolChain &TC) : gnutools::Assembler(TC) {
    DefaultAssembler = "gas";
  }

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("solaris::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace solaris
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY Solaris : public Generic_ELF {
public:
  friend class tools::solaris::Linker;

  struct LinkerChoice {
    bool IsGnuLd;
    StringRef PathToLinker;
  };

  Solaris(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  void
  addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;

  /// TODO: test if newer versions of the Solaris linker
  /// respect --as--needed/--no-as-needed. If so, this
  /// override can be removed.
  void addAsNeededOption(llvm::opt::ArgStringList &CmdArgs,
                         bool as_needed) const override;

  SanitizerMask getSupportedSanitizers() const override;

  const char *getDefaultLinker() const override;

  virtual bool mustElideDynamicList() const override;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;

private:
  bool isLinkerGnuLd() const;
  LinkerChoice chooseLinker() const;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SOLARIS_H
