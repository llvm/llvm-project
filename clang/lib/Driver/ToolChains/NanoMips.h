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

class LLVM_LIBRARY_VISIBILITY NanoMips : public Generic_ELF {
 public:
  NanoMips(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args);

  void
    AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                              llvm::opt::ArgStringList &CC1Args) const override;
};

}
}
}

#endif
