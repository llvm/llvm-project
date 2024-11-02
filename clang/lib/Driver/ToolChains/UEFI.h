//===--- UEFI.h - UEFI ToolChain Implementations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_UEFI_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_UEFI_H

#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang::driver {
namespace tools {
namespace uefi {
class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("uefi::Linker", "lld-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace uefi
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY UEFI : public ToolChain {
public:
  UEFI(const Driver &D, const llvm::Triple &Triple,
       const llvm::opt::ArgList &Args);

protected:
  Tool *buildLinker() const override;

public:
  bool HasNativeLLVMSupport() const override { return true; }
  UnwindTableLevel
  getDefaultUnwindTableLevel(const llvm::opt::ArgList &Args) const override {
    return UnwindTableLevel::Asynchronous;
  }
  bool isPICDefault() const override { return true; }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override {
    return false;
  }
  bool isPICDefaultForced() const override { return true; }
};

} // namespace toolchains
} // namespace clang::driver

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_UEFI_H
