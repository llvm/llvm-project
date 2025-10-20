//===-- LFILinux.cpp - LFI ToolChain Implementations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LFILinux.h"

using namespace clang::driver::toolchains;
using namespace llvm::opt;

void LFILinuxToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                            ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    if (Args.hasArg(options::OPT_fexperimental_library))
      CmdArgs.push_back("-lc++experimental");
    CmdArgs.push_back("-lc++abi");
    break;
  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}
