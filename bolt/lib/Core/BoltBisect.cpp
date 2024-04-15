//===- bolt/Bisect.cpp - LLVM Bisect support -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements support for a bisecting optimizations based on a
/// command line option.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BoltBisect.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

static BoltBisect &getBoltBisector() {
  static BoltBisect BoltBisector;
  return BoltBisector;
}

static cl::opt<int> BoltBisectLimit("opt-bisect-limit", cl::Hidden,
                                   cl::init(BoltBisect::Disabled), cl::Optional,
                                   cl::cb<void, int>([](int Limit) {
                                     getBoltBisector().setLimit(Limit);
                                   }),
                                   cl::desc("Maximum optimization to perform"));

static void printPassMessage(const StringRef &Name, int PassNum,
                             bool Running) {
  StringRef Status = Running ? "" : "NOT ";
  errs() << "BISECT: " << Status << "running pass "
         << "(" << PassNum << ") " << Name << " on " << "\n";
}

bool BoltBisect::shouldRunPass(const StringRef PassName) {
  assert(isEnabled());

  int CurBisectNum = ++LastBisectNum;
  bool ShouldRun = (BisectLimit == -1 || CurBisectNum <= BisectLimit);
  printPassMessage(PassName, CurBisectNum, ShouldRun);
  return ShouldRun;
}

const int BoltBisect::Disabled;

BoltPassGate &llvm::getGlobalPassGate() { return getBoltBisector(); }
