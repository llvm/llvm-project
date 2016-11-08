//===--- NonCopyableObjects.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NonCopyableObjects.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void NonCopyableObjectsCheck::registerMatchers(MatchFinder *Finder) {
  // There are two ways to get into trouble with objects like FILE *:
  // dereferencing the pointer type to be a non-pointer type, and declaring
  // the type as a non-pointer type in the first place. While the declaration
  // itself could technically be well-formed in the case where the type is not
  // an opaque type, it's highly suspicious behavior.
  //
  // POSIX types are a bit different in that it's reasonable to declare a
  // non-pointer variable or data member of the type, but it is not reasonable
  // to dereference a pointer to the type, or declare a parameter of non-pointer
  // type.
  // FIXME: it would be good to make a list that is also user-configurable so
  // that users can add their own elements to the list. However, it may require
  // some extra thought since POSIX types and FILE types are usable in different
  // ways.

  auto BadFILEType = hasType(
      namedDecl(hasAnyName("::FILE", "FILE", "std::FILE")).bind("type_decl"));
  auto BadPOSIXType =
      hasType(namedDecl(hasAnyName("::pthread_cond_t", "::pthread_mutex_t",
                                   "pthread_cond_t", "pthread_mutex_t"))
                  .bind("type_decl"));
  auto BadEitherType = anyOf(BadFILEType, BadPOSIXType);

  Finder->addMatcher(
      namedDecl(anyOf(varDecl(BadFILEType), fieldDecl(BadFILEType)))
          .bind("decl"),
      this);
  Finder->addMatcher(parmVarDecl(BadPOSIXType).bind("decl"), this);
  Finder->addMatcher(
      expr(unaryOperator(hasOperatorName("*"), BadEitherType)).bind("expr"),
      this);
}

void NonCopyableObjectsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<NamedDecl>("decl");
  const auto *BD = Result.Nodes.getNodeAs<NamedDecl>("type_decl");
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");

  if (D && BD)
    diag(D->getLocation(), "%0 declared as type '%1', which is unsafe to copy"
                           "; did you mean '%1 *'?")
        << D << BD->getName();
  else if (E)
    diag(E->getExprLoc(),
         "expression has opaque data structure type %0; type should only be "
         "used as a pointer and not dereferenced")
        << BD;
}

} // namespace misc
} // namespace tidy
} // namespace clang

