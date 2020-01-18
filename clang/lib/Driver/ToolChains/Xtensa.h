//===--- Xtensa.h - Xtensa Tool and ToolChain Implementations ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_Xtensa_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_Xtensa_H

#include "Gnu.h"
#include "InputInfo.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace toolchains {

class XtensaGCCToolchainDetector {
public:
  std::string GCCLibAndIncVersion;
  std::string GCCToolchainName;
  std::string GCCToolchainDir;

  XtensaGCCToolchainDetector(const Driver &D, const llvm::Triple &HostTriple,
                             const llvm::opt::ArgList &Args);

  bool IsValid() const { return GCCToolchainName != ""; }
};

class LLVM_LIBRARY_VISIBILITY XtensaToolChain : public Generic_ELF {
protected:
  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;

public:
  XtensaToolChain(const Driver &D, const llvm::Triple &Triple,
                  const llvm::opt::ArgList &Args);
  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void
  addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  bool IsIntegratedAssemblerDefault() const override {
    return (IsIntegratedAsm || (XtensaGCCToolchain.GCCToolchainName == ""));
  }

  static const StringRef GetTargetCPUVersion(const llvm::opt::ArgList &Args);

  XtensaGCCToolchainDetector XtensaGCCToolchain;
  bool IsIntegratedAsm = true;
};

} // end namespace toolchains

namespace tools {
namespace Xtensa {
class LLVM_LIBRARY_VISIBILITY Linker : public GnuTool {
public:
  Linker(const ToolChain &TC)
      : GnuTool("Xtensa::Linker", "xtensa-esp32-elf-ld", TC) {}
  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC)
      : Tool("Xtensa::Assembler", "xtensa-esp32-elf-as", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace Xtensa
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_Xtensa_H
