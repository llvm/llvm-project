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
      hasType(pointsTo(cxxRecordDecl(isNotTriviallyCopyable())));

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
  if (const auto *Destination = Result.Nodes.getNodeAs<CallExpr>("dest")) {
    diag(Destination->getLocStart(), "undefined behavior, destination "
                                     "object is not TriviallyCopyable");
  }
  if (const auto *Source = Result.Nodes.getNodeAs<CallExpr>("src")) {
    diag(Source->getLocStart(), "undefined behavior, source object is not "
                                "TriviallyCopyable");
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
