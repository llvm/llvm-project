//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeStaticCastDowncastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

ProTypeStaticCastDowncastCheck::ProTypeStaticCastDowncastCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", true)) {}

void ProTypeStaticCastDowncastCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

void ProTypeStaticCastDowncastCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxStaticCastExpr(hasCastKind(CK_BaseToDerived)).bind("cast"), this);
}

void ProTypeStaticCastDowncastCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CXXStaticCastExpr>("cast");

  QualType SourceType = MatchedCast->getSubExpr()->getType();
  const auto *SourceDecl = SourceType->getPointeeCXXRecordDecl();
  if (!SourceDecl) // The cast is from object to reference
    SourceDecl = SourceType->getAsCXXRecordDecl();
  if (!SourceDecl)
    return;

  if (SourceDecl->isPolymorphic()) {
    diag(MatchedCast->getOperatorLoc(),
         "do not use static_cast to downcast from a base to a derived class; "
         "use dynamic_cast instead")
        << FixItHint::CreateReplacement(MatchedCast->getOperatorLoc(),
                                        "dynamic_cast");
    return;
  }

  if (!StrictMode)
    return;

  diag(MatchedCast->getOperatorLoc(),
       "do not use static_cast to downcast from a base to a derived class");
}

} // namespace clang::tidy::cppcoreguidelines
