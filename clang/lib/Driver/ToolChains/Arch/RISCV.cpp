//===--- RISCV.cpp - RISCV Helpers for Tools --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

void riscv::getRISCVTargetFeatures(const Driver &D, const ArgList &Args,
                                   std::vector<StringRef> &Features) {
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    StringRef MArch = A->getValue();
    // TODO: handle rv64
    std::pair<StringRef, StringRef> MArchSplit = StringRef(MArch).split("rv32");
    if (!MArchSplit.second.size())
      return;

    for (char c : MArchSplit.second) {
      switch (c) {
      case 'i':
        break;
      case 'm':
        Features.push_back("+m");
        break;
      case 'a':
        Features.push_back("+a");
        break;
      case 'f':
        Features.push_back("+f");
        break;
      case 'd':
        Features.push_back("+d");
        break;
      case 'c':
        Features.push_back("+c");
        break;
      }
    }
  }
}

StringRef riscv::getRISCVABI(const ArgList &Args, const llvm::Triple &Triple) {
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    return A->getValue();

  return Triple.getArch() == llvm::Triple::riscv32 ? "ilp32" : "lp64";
}
