//== UnsignedStatTesterChecker.cpp --------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker demonstrates the use of UnsignedEPStat for per-entry-point
// statistics. It conditionally sets a statistic based on the entry point name.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/EntryPointStats.h"

using namespace clang;
using namespace ento;

#define DEBUG_TYPE "UnsignedStatTester"

static UnsignedEPStat DemoStat("DemoStat");

namespace {
class UnsignedStatTesterChecker : public Checker<check::BeginFunction> {
public:
  void checkBeginFunction(CheckerContext &C) const;
};
} // namespace

void UnsignedStatTesterChecker::checkBeginFunction(CheckerContext &C) const {
  std::string Name;
  if (const Decl *D = C.getLocationContext()->getDecl())
    if (const FunctionDecl *F = D->getAsFunction())
      Name = F->getNameAsString();

  // Conditionally set the statistic based on the function name
  if (Name == "func_one") {
    DemoStat.set(1);
  } else if (Name == "func_two") {
    DemoStat.set(2);
  } else if (Name == "func_three") {
    DemoStat.set(3);
  }
  // For any other function (e.g., "func_none"), don't set the statistic
}

void ento::registerUnsignedStatTesterChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UnsignedStatTesterChecker>();
}

bool ento::shouldRegisterUnsignedStatTesterChecker(const CheckerManager &) {
  return true;
}
