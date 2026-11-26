//== CheckerContext.cpp - Context info for path-sensitive checkers-----------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerContext that provides contextual info for
//  path-sensitive checkers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/Lex/Lexer.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"

using namespace clang;
using namespace ento;

const FunctionDecl *CheckerContext::getCalleeDecl(const CallExpr *CE) const {
  const FunctionDecl *D = CE->getDirectCallee();
  if (D)
    return D;

  const Expr *Callee = CE->getCallee();
  SVal L = Pred->getSVal(Callee);
  return L.getAsFunctionDecl();
}

StringRef CheckerContext::getCalleeName(const FunctionDecl *FunDecl) const {
  if (!FunDecl)
    return StringRef();
  IdentifierInfo *funI = FunDecl->getIdentifier();
  if (!funI)
    return StringRef();
  return funI->getName();
}

StringRef CheckerContext::getDeclDescription(const Decl *D) {
  if (isa<ObjCMethodDecl, CXXMethodDecl>(D))
    return "method";
  if (isa<BlockDecl>(D))
    return "anonymous block";
  return "function";
}

bool CheckerContext::isCLibraryFunction(const FunctionDecl *FD,
                                        StringRef Name) {
  return CLibraryFunctionInfo(FD).isCLibraryFunction(Name);
}

bool CheckerContext::isHardenedVariantOf(const FunctionDecl *FD,
                                         StringRef Name) {
  const IdentifierInfo *II = FD->getIdentifier();
  if (!II)
    return false;

  auto CompletelyMatchesParts = [II](auto... Parts) -> bool {
    StringRef FName = II->getName();
    return (FName.consume_front(Parts) && ...) && FName.empty();
  };

  return CompletelyMatchesParts("__", Name, "_chk") ||
         CompletelyMatchesParts("__builtin_", "__", Name, "_chk");
}

std::string CheckerContext::getMacroNameOrSpelling(SourceLocation &Loc) {
  const auto &SM = getSourceManager();
  const auto &LO = getLangOpts();
  if (Loc.isMacroID())
    return Lexer::getImmediateMacroName(Loc, SM, LO).str();
  llvm::SmallString<16> Buf;
  return Lexer::getSpelling(Loc, Buf, SM, LO).str();
}

/// Evaluate comparison and return true if it's known that condition is true
static bool evalComparison(SVal LHSVal, BinaryOperatorKind ComparisonOp,
                           SVal RHSVal, ProgramStateRef State) {
  if (LHSVal.isUnknownOrUndef())
    return false;
  ProgramStateManager &Mgr = State->getStateManager();
  if (!isa<NonLoc>(LHSVal)) {
    LHSVal = Mgr.getStoreManager().getBinding(State->getStore(),
                                              LHSVal.castAs<Loc>());
    if (LHSVal.isUnknownOrUndef() || !isa<NonLoc>(LHSVal))
      return false;
  }

  SValBuilder &Bldr = Mgr.getSValBuilder();
  SVal Eval = Bldr.evalBinOp(State, ComparisonOp, LHSVal, RHSVal,
                             Bldr.getConditionType());
  if (Eval.isUnknownOrUndef())
    return false;
  ProgramStateRef StTrue, StFalse;
  std::tie(StTrue, StFalse) = State->assume(Eval.castAs<DefinedSVal>());
  return StTrue && !StFalse;
}

bool CheckerContext::isGreaterOrEqual(const Expr *E, unsigned long long Val) {
  DefinedSVal V = getSValBuilder().makeIntVal(Val, getASTContext().LongLongTy);
  return evalComparison(getSVal(E), BO_GE, V, getState());
}

bool CheckerContext::isNegative(const Expr *E) {
  DefinedSVal V = getSValBuilder().makeIntVal(0, false);
  return evalComparison(getSVal(E), BO_LT, V, getState());
}
