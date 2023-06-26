//===--- Gnu.cpp - Gnu Tool and ToolChain Implementations -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_NANOMIPS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_NANOMIPS_H

#include "Gnu.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Support/Debug.h"

namespace clang {
namespace driver {
namespace toolchains {

class LLVM_LIBRARY_VISIBILITY NanoMipsLinker : public tools::gnutools::Linker {
public:
  NanoMipsLinker( const ToolChain &TC) : Linker(TC) { }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

};

class LLVM_LIBRARY_VISIBILITY NanoMips : public Generic_ELF {
 public:
  NanoMips(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args);

  void
    AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                              llvm::opt::ArgStringList &CC1Args) const override;

  Tool *buildLinker() const override {
    return new NanoMipsLinker(*this);
  }

  UnwindLibType GetUnwindLibType(const llvm::opt::ArgList &Args) const override {
    return ToolChain::UNW_None;
  }

  bool HasNativeLLVMSupport() const override {
    // Not strictly true, but necessary for LTO.
    return true;
  }

  bool useIntegratedAs() const override {
    // No integrated assembler for NanoMips
    return false;
  }

};

} // toolchains
} // driver
} // clang

#endif
