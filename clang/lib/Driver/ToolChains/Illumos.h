//===--- Illumos.h - Illumos ToolChain Implementations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ILLUMOS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ILLUMOS_H

#include "Gnu.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace tools {

namespace illumos {
class LLVM_LIBRARY_VISIBILITY Assembler final : public gnutools::Assembler {
public:
  Assembler(const ToolChain &TC) : gnutools::Assembler(TC) {
    DefaultAssembler = "gas";
  }

  bool hasIntegratedCPP() const override { return false; }
};

class LLVM_LIBRARY_VISIBILITY Linker final : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("illumos::Linker", "linker", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  std::string getLinkerPath(const llvm::opt::ArgList &Args) const;

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace illumos
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY Illumos : public Generic_ELF {
public:
  Illumos(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  void
  addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;

  /// The Illumos linker supports an equivalent of --as-needed/--no-as-needed.
  void addAsNeededOption(llvm::opt::ArgStringList &CmdArgs,
                         bool as_needed) const override;

  SanitizerMask getSupportedSanitizers() const override;

  const char *getDefaultLinker() const override;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ILLUMOS_H
