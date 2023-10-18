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

namespace {
AST_MATCHER(QualType, hasConst) { return hasConstQualifier(Node); }
} // namespace

ProTypeConstCastCheck::ProTypeConstCastCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)) {}

void ProTypeConstCastCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

void ProTypeConstCastCheck::registerMatchers(MatchFinder *Finder) {
  if (StrictMode)
    Finder->addMatcher(cxxConstCastExpr().bind("cast"), this);
  else
    Finder->addMatcher(cxxConstCastExpr(unless(hasDestinationType(
                                            hasCanonicalType(hasConst()))))
                           .bind("cast"),
                       this);
}

void ProTypeConstCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CXXConstCastExpr>("cast");
  diag(MatchedCast->getOperatorLoc(),
       "do not use const_cast%select{ to cast away const|}0")
      << StrictMode;
}

} // namespace clang::tidy::cppcoreguidelines
