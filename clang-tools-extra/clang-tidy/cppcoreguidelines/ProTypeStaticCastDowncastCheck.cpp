//===--- ProTypeStaticCastDowncastCheck.cpp - clang-tidy-------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProTypeStaticCastDowncastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void ProTypeStaticCastDowncastCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      cxxStaticCastExpr(unless(isInTemplateInstantiation())).bind("cast"),
      this);
}

void ProTypeStaticCastDowncastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CXXStaticCastExpr>("cast");
  if (MatchedCast->getCastKind() != CK_BaseToDerived)
    return;

  QualType SourceType = MatchedCast->getSubExpr()->getType();
  const auto *SourceDecl = SourceType->getPointeeCXXRecordDecl();
  if (!SourceDecl) // The cast is from object to reference
    SourceDecl = SourceType->getAsCXXRecordDecl();
  if (!SourceDecl)
    return;

  if (SourceDecl->isPolymorphic())
    diag(MatchedCast->getOperatorLoc(),
         "do not use static_cast to downcast from a base to a derived class; "
         "use dynamic_cast instead")
        << FixItHint::CreateReplacement(MatchedCast->getOperatorLoc(),
                                        "dynamic_cast");
  else
    diag(MatchedCast->getOperatorLoc(),
         "do not use static_cast to downcast from a base to a derived class");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
