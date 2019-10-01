//===--- Contiki.cpp - Contiki ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Contiki.h"
#include "CommonArgs.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

Contiki::Contiki(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {}

SanitizerMask Contiki::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  if (IsX86)
    Res |= SanitizerKind::SafeStack;
  return Res;
}
