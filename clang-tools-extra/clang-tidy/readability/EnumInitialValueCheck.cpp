//===--- EnumInitialValueCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EnumInitialValueCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallString.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER(EnumDecl, isNoneEnumeratorsInitialized) {
  return llvm::all_of(Node.enumerators(), [](EnumConstantDecl const *ECD) {
    return ECD->getInitExpr() == nullptr;
  });
}

AST_MATCHER(EnumDecl, isOnlyFirstEnumeratorsInitialized) {
  for (EnumConstantDecl const *ECD : Node.enumerators())
    if (ECD == *Node.enumerator_begin()) {
      if (ECD->getInitExpr() == nullptr)
        return false;
    } else {
      if (ECD->getInitExpr() != nullptr)
        return false;
    }
  return true;
}

AST_MATCHER(EnumDecl, isAllEnumeratorsInitialized) {
  return llvm::all_of(Node.enumerators(), [](EnumConstantDecl const *ECD) {
    return ECD->getInitExpr() != nullptr;
  });
}

} // namespace

void EnumInitialValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(enumDecl(unless(anyOf(isNoneEnumeratorsInitialized(),
                                           isOnlyFirstEnumeratorsInitialized(),
                                           isAllEnumeratorsInitialized())))
                         .bind("enum"),
                     this);
}

void EnumInitialValueCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("enum");
  assert(Enum != nullptr);
  SourceLocation Loc = Enum->getBeginLoc();
  if (Loc.isInvalid() || Loc.isMacroID())
    return;
  DiagnosticBuilder Diag =
      diag(Loc, "inital values in enum %0 are not consistent, consider "
                "explicit initialization first, all or none of enumerators")
      << Enum->getName();
  for (EnumConstantDecl const *ECD : Enum->enumerators())
    if (ECD->getInitExpr() == nullptr) {
      SourceLocation ECDLoc = ECD->getEndLoc();
      if (ECDLoc.isInvalid() || ECDLoc.isMacroID())
        continue;
      std::optional<Token> Next = utils::lexer::findNextTokenSkippingComments(
          ECDLoc, *Result.SourceManager, getLangOpts());
      if (!Next.has_value() || Next->getLocation().isMacroID())
        continue;
      llvm::SmallString<8> Str{" = "};
      ECD->getInitVal().toString(Str);
      Diag << FixItHint::CreateInsertion(Next->getLocation(), Str);
    }
}

} // namespace clang::tidy::readability
