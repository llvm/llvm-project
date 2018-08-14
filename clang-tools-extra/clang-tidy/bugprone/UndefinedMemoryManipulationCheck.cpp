//===--- UndefinedMemoryManipulationCheck.cpp - clang-tidy-----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UndefinedMemoryManipulationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
AST_MATCHER(CXXRecordDecl, isNotTriviallyCopyable) {
  // For incomplete types, assume they are TriviallyCopyable.
  return Node.hasDefinition() ? !Node.isTriviallyCopyable() : false;
}
} // namespace

void UndefinedMemoryManipulationCheck::registerMatchers(MatchFinder *Finder) {
  const auto NotTriviallyCopyableObject =
      hasType(ast_matchers::hasCanonicalType(
          pointsTo(cxxRecordDecl(isNotTriviallyCopyable()))));

  // Check whether destination object is not TriviallyCopyable.
  // Applicable to all three memory manipulation functions.
  Finder->addMatcher(callExpr(callee(functionDecl(hasAnyName(
                                  "::memset", "::memcpy", "::memmove"))),
                              hasArgument(0, NotTriviallyCopyableObject))
                         .bind("dest"),
                     this);

  // Check whether source object is not TriviallyCopyable.
  // Only applicable to memcpy() and memmove().
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName("::memcpy", "::memmove"))),
               hasArgument(1, NotTriviallyCopyableObject))
          .bind("src"),
      this);
}

void UndefinedMemoryManipulationCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("dest")) {
    QualType DestType = Call->getArg(0)->IgnoreImplicit()->getType();
    if (!DestType->getPointeeType().isNull())
      DestType = DestType->getPointeeType();
    diag(Call->getBeginLoc(), "undefined behavior, destination object type %0 "
                              "is not TriviallyCopyable")
        << DestType;
  }
  if (const auto *Call = Result.Nodes.getNodeAs<CallExpr>("src")) {
    QualType SourceType = Call->getArg(1)->IgnoreImplicit()->getType();
    if (!SourceType->getPointeeType().isNull())
      SourceType = SourceType->getPointeeType();
    diag(Call->getBeginLoc(),
         "undefined behavior, source object type %0 is not TriviallyCopyable")
        << SourceType;
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
