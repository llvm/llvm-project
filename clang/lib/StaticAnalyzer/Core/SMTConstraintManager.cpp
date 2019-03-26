//== SMTConstraintManager.cpp -----------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConstraintManager.h"

using namespace clang;
using namespace ento;

std::unique_ptr<ConstraintManager>
ento::CreateZ3ConstraintManager(ProgramStateManager &StMgr, SubEngine *Eng) {
  return llvm::make_unique<SMTConstraintManager>(Eng, StMgr.getSValBuilder());
}
