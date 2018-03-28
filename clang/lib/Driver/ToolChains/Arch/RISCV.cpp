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
    if (!(MArch.startswith("rv32") || MArch.startswith("rv64")) ||
        (MArch.size() < 5)) {
      // ISA string must begin with rv32 or rv64.
      // TODO: Improve diagnostic message.
      D.Diag(diag::err_drv_invalid_arch_name) << MArch;
      return;
    }

    // The canonical order specified in ISA manual.
    // Ref: Table 22.1 in RISC-V User-Level ISA V2.2
    StringRef StdExts = "mafdc";

    bool HasF = false, HasD = false;
    char Baseline = MArch[4];

    // TODO: Add 'e' once backend supported.
    switch (Baseline) {
    default:
      // First letter should be 'e', 'i' or 'g'.
      // TODO: Improve diagnostic message.
      D.Diag(diag::err_drv_invalid_arch_name) << MArch;
      return;
    case 'i':
      break;
    case 'g':
      // g = imafd
      StdExts = StdExts.drop_front(4);
      Features.push_back("+m");
      Features.push_back("+a");
      Features.push_back("+f");
      Features.push_back("+d");
      HasF = true;
      HasD = true;
      break;
    }

    auto StdExtsItr = StdExts.begin();
    // Skip rvxxx
    StringRef Exts = MArch.substr(5);

    for (char c : Exts) {
      // Check ISA extensions are specified in the canonical order.
      while (StdExtsItr != StdExts.end() && *StdExtsItr != c)
        ++StdExtsItr;

      if (StdExtsItr == StdExts.end()) {
        // TODO: Improve diagnostic message.
        D.Diag(diag::err_drv_invalid_arch_name) << MArch;
        return;
      }

      // Move to next char to prevent repeated letter.
      ++StdExtsItr;

      // The order is OK, then push it into features.
      switch (c) {
      default:
        // TODO: Improve diagnostic message.
        D.Diag(diag::err_drv_invalid_arch_name) << MArch;
        return;
      case 'm':
        Features.push_back("+m");
        break;
      case 'a':
        Features.push_back("+a");
        break;
      case 'f':
        Features.push_back("+f");
        HasF = true;
        break;
      case 'd':
        Features.push_back("+d");
        HasD = true;
        break;
      case 'c':
        Features.push_back("+c");
        break;
      }
    }

    // Dependency check
    // It's illegal to specify the 'd' (double-precision floating point)
    // extension without also specifying the 'f' (single precision
    // floating-point) extension.
    // TODO: Improve diagnostic message.
    if (HasD && !HasF)
      D.Diag(diag::err_drv_invalid_arch_name) << MArch;
  }
}

StringRef riscv::getRISCVABI(const ArgList &Args, const llvm::Triple &Triple) {
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    return A->getValue();

  return Triple.getArch() == llvm::Triple::riscv32 ? "ilp32" : "lp64";
}
