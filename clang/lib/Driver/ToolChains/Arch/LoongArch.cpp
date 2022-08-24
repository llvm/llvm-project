//===--- LoongArch.cpp - LoongArch Helpers for Tools ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"

using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

StringRef loongarch::getLoongArchABI(const ArgList &Args,
                                     const llvm::Triple &Triple) {
  assert((Triple.getArch() == llvm::Triple::loongarch32 ||
          Triple.getArch() == llvm::Triple::loongarch64) &&
         "Unexpected triple");

  // If `-mabi=` is specified, use it.
  if (const Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    return A->getValue();

  // Choose a default based on the triple.
  // TODO: select appropiate ABI.
  return Triple.getArch() == llvm::Triple::loongarch32 ? "ilp32d" : "lp64d";
}
