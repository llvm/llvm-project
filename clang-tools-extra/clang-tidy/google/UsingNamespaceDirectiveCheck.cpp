//===--- UsingNamespaceDirectiveCheck.cpp - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UsingNamespaceDirectiveCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace build {

void UsingNamespaceDirectiveCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
    Finder->addMatcher(usingDirectiveDecl().bind("usingNamespace"), this);
}

void UsingNamespaceDirectiveCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *U = Result.Nodes.getNodeAs<UsingDirectiveDecl>("usingNamespace");
  SourceLocation Loc = U->getBeginLoc();
  if (U->isImplicit() || !Loc.isValid())
    return;

  // Do not warn if namespace is a std namespace with user-defined literals. The
  // user-defined literals can only be used with a using directive.
  if (isStdLiteralsNamespace(U->getNominatedNamespace()))
    return;

  diag(Loc, "do not use namespace using-directives; "
            "use using-declarations instead");
  // TODO: We could suggest a list of using directives replacing the using
  //       namespace directive.
}

bool UsingNamespaceDirectiveCheck::isStdLiteralsNamespace(
    const NamespaceDecl *NS) {
  if (!NS->getName().endswith("literals"))
    return false;

  const auto *Parent = dyn_cast_or_null<NamespaceDecl>(NS->getParent());
  if (!Parent)
    return false;

  if (Parent->isStdNamespace())
    return true;

  return Parent->getName() == "literals" && Parent->getParent() &&
         Parent->getParent()->isStdNamespace();
}
} // namespace build
} // namespace google
} // namespace tidy
} // namespace clang
