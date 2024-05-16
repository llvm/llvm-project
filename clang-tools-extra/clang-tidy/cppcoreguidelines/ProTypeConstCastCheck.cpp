//===--- ProTypeConstCastCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeConstCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

static bool hasConstQualifier(QualType Type) {
  const QualType PtrType = Type->getPointeeType();
  if (!PtrType.isNull())
    return hasConstQualifier(PtrType);

  return Type.isConstQualified();
}

static bool hasVolatileQualifier(QualType Type) {
  const QualType PtrType = Type->getPointeeType();
  if (!PtrType.isNull())
    return hasVolatileQualifier(PtrType);
  return Type.isVolatileQualified();
}

ProTypeConstCastCheck::ProTypeConstCastCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)) {}

void ProTypeConstCastCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

void ProTypeConstCastCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstCastExpr().bind("cast"), this);
}

void ProTypeConstCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CXXConstCastExpr>("cast");
  if (StrictMode) {
    diag(MatchedCast->getOperatorLoc(), "do not use const_cast");
    return;
  }

  const QualType TargetType = MatchedCast->getType().getCanonicalType();
  const QualType SourceType =
      MatchedCast->getSubExpr()->getType().getCanonicalType();

  const bool RemovingConst =
      hasConstQualifier(SourceType) && !hasConstQualifier(TargetType);
  const bool RemovingVolatile =
      hasVolatileQualifier(SourceType) && !hasVolatileQualifier(TargetType);

  if (!RemovingConst && !RemovingVolatile) {
    // Cast is doing nothing.
    return;
  }

  diag(MatchedCast->getOperatorLoc(),
       "do not use const_cast to remove%select{| const}0%select{| "
       "and}2%select{| volatile}1 qualifier")
      << RemovingConst << RemovingVolatile
      << (RemovingConst && RemovingVolatile);
}

} // namespace clang::tidy::cppcoreguidelines
