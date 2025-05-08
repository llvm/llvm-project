//===-- NullPointerAnalysisModel.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/DeadCodeModel.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace clang::dataflow {
namespace {
using namespace ast_matchers;
using Diagnoser = DeadCodeDiagnoser;

constexpr char kCond[] = "condition";

auto conditionMatcher() {
  return expr(allOf(hasType(booleanType()), unless(cxxBoolLiteral())))
      .bind(kCond);
}

DeadCodeDiagnoser::ResultType
diagnoseAnyDeadCondition(const Expr *S, const MatchFinder::MatchResult &Result,
                         const Environment &Env) {
  const auto *Cond = Result.Nodes.getNodeAs<Expr>(kCond);
  assert(Cond != nullptr);

  Value *Val = Env.getValue(*Cond);
  if (!Val || !Val->getProperty(kCond))
    return {};

  if (Env.allows(Env.getBoolLiteralValue(false).formula())) {
    // We are already in a dead code section, bail early
    return {};
  }

  if (BoolValue *CondValue = cast_or_null<BoolValue>(Env.getValue(*Cond))) {
    if (Env.proves(CondValue->formula())) {
      return {{Cond->getBeginLoc(), Diagnoser::DiagnosticType::AlwaysTrue}};
    }

    if (Env.proves(Env.arena().makeNot(CondValue->formula()))) {
      return {{Cond->getBeginLoc(), Diagnoser::DiagnosticType::AlwaysFalse}};
    }
  }

  return {};
}

DeadCodeDiagnoser::ResultType catchall(const Stmt *S,
                                       const MatchFinder::MatchResult &Result,
                                       const Environment &Env) {
  S->dump();
  return {};
}

auto buildDiagnoseMatchSwitch() {
  return CFGMatchSwitchBuilder<const Environment, Diagnoser::ResultType>()
      .CaseOfCFGStmt<Expr>(conditionMatcher(), diagnoseAnyDeadCondition)
      .Build();
}

} // namespace

void DeadCodeModel::transferBranch(bool Branch, const Stmt *S,
                                   NoopLattice &State, Environment &Env) {
  if (!S || !isa<Expr>(S))
    return;

  Value *Val = Env.getValue(*cast<Expr>(S));

  if (!Val)
    return;

  Val->setProperty(kCond, Env.getBoolLiteralValue(true));
}

DeadCodeDiagnoser::DeadCodeDiagnoser()
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch()) {}

DeadCodeDiagnoser::ResultType DeadCodeDiagnoser::operator()(
    const CFGElement &Elt, ASTContext &Ctx,
    const TransferStateForDiagnostics<NoopLattice> &State) {
  return DiagnoseMatchSwitch(Elt, Ctx, State.Env);
}

} // namespace clang::dataflow