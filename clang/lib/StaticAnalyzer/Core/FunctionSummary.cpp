//== FunctionSummary.h - Stores summaries of functions. ------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a summary of a function gathered/used by static analyzes.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/FunctionSummary.h"
using namespace clang;
using namespace ento;

FunctionSummariesTy::~FunctionSummariesTy() {
  for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I) {
    delete(I->second);
  }
}

unsigned FunctionSummariesTy::getTotalNumBasicBlocks() {
  unsigned Total = 0;
  for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I) {
    Total += I->second->TotalBasicBlocks;
  }
  return Total;
}

unsigned FunctionSummariesTy::getTotalNumVisitedBasicBlocks() {
  unsigned Total = 0;
  for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I) {
    Total += I->second->VisitedBasicBlocks.count();
  }
  return Total;
}
