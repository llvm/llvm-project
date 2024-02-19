//===--- ReferenceToConstructedTemporaryCheck.cpp - clang-tidy
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReferenceToConstructedTemporaryCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

// Predicate structure to check if lifetime of temporary is not extended by
// ValueDecl pointed out by ID
struct NotExtendedByDeclBoundToPredicate {
  bool operator()(const internal::BoundNodesMap &Nodes) const {
    const auto *Other = Nodes.getNodeAs<ValueDecl>(ID);
    if (!Other)
      return true;

    const auto *Self = Node.get<MaterializeTemporaryExpr>();
    if (!Self)
      return true;

    return Self->getExtendingDecl() != Other;
  }

  StringRef ID;
  ::clang::DynTypedNode Node;
};

AST_MATCHER_P(MaterializeTemporaryExpr, isExtendedByDeclBoundTo, StringRef,
              ID) {
  NotExtendedByDeclBoundToPredicate Predicate{
      ID, ::clang::DynTypedNode::create(Node)};
  return Builder->removeBindings(Predicate);
}

} // namespace

bool ReferenceToConstructedTemporaryCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

std::optional<TraversalKind>
ReferenceToConstructedTemporaryCheck::getCheckTraversalKind() const {
  return TK_AsIs;
}

void ReferenceToConstructedTemporaryCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      varDecl(unless(isExpansionInSystemHeader()),
              hasType(qualType(references(qualType().bind("type")))),
              decl().bind("var"),
              hasInitializer(expr(hasDescendant(
                  materializeTemporaryExpr(
                      isExtendedByDeclBoundTo("var"),
                      has(expr(anyOf(cxxTemporaryObjectExpr(), initListExpr(),
                                     cxxConstructExpr()),
                               hasType(qualType(equalsBoundNode("type"))))))
                      .bind("temporary"))))),
      this);
}

void ReferenceToConstructedTemporaryCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *MatchedTemporary = Result.Nodes.getNodeAs<Expr>("temporary");

  diag(MatchedDecl->getLocation(),
       "reference variable %0 extends the lifetime of a just-constructed "
       "temporary object %1, consider changing reference to value")
      << MatchedDecl << MatchedTemporary->getType();
}

} // namespace clang::tidy::readability
