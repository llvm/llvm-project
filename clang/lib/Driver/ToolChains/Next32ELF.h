//===--- Next32ELF.h - Next32ELF ToolChain Implementations ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_NEXT32_ELF_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_NEXT32_ELF_H

#include "Gnu.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace tools {
namespace Next32 {

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("Next32::Linker", "Next32-lld", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace Next32
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY Next32LLVMToolChain : public Generic_ELF {
protected:
  Tool *buildLinker() const override;

public:
  Next32LLVMToolChain(const Driver &D, const llvm::Triple &Triple,
                      const llvm::opt::ArgList &Args);

  unsigned GetDefaultDwarfVersion() const override;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;

  const char *getDefaultLinker() const override;

  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  CXXStdlibType GetDefaultCXXStdlibType() const override;

private:
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_NEXT32_ELF_H
