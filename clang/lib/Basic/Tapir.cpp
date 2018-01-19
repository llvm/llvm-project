//===--- Tapir.cpp - C Language Family Language Options ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the functions from Tapir.h
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Tapir.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

TapirTargetType clang::parseTapirTarget(const ArgList &Args) {
  // Use Cilk if -ftapir is not specified but either -fcilkplus or -fdetach is
  // specified.
  if (!Args.hasArg(options::OPT_ftapir_EQ)) {
    if (Args.hasArg(options::OPT_fcilkplus) ||
        Args.hasArg(options::OPT_fdetach))
      return TapirTargetType::Cilk;
    return TapirTargetType::None;
  }

  // Otherwise use the runtime specified by -ftapir.
  TapirTargetType TapirTarget = TapirTargetType::None;
  if (const Arg *A = Args.getLastArg(options::OPT_ftapir_EQ))
    TapirTarget = llvm::StringSwitch<TapirTargetType>(A->getValue())
      .Case("serial", TapirTargetType::Serial)
      .Case("cilk", TapirTargetType::Cilk)
      .Case("openmp", TapirTargetType::OpenMP)
      .Case("cilkr", TapirTargetType::CilkR)
      .Default(TapirTargetType::None);

  return TapirTarget;
}
