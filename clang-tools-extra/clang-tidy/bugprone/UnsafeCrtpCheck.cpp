//===--- UnsafeCrtpCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeCrtpCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
// Finds a node if it's already a bound node.
AST_MATCHER_P(CXXRecordDecl, isBoundNode, std::string, ID) {
  return Builder->removeBindings(
      [&](const ast_matchers::internal::BoundNodesMap &Nodes) {
        const auto *BoundRecord = Nodes.getNodeAs<CXXRecordDecl>(ID);
        return BoundRecord != &Node;
      });
}
} // namespace

void UnsafeCrtpCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(classTemplateSpecializationDecl(
                         decl().bind("crtp"),
                         hasAnyTemplateArgument(refersToType(recordType(
                             hasDeclaration(cxxRecordDecl(isDerivedFrom(
                                 cxxRecordDecl(isBoundNode("crtp"))))))))),
                     this);
}

void UnsafeCrtpCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCRTP = Result.Nodes.getNodeAs<CXXRecordDecl>("crtp");

  MatchedCRTP->dump();

  for (auto &&Ctor : MatchedCRTP->ctors()) {
    if (Ctor->getAccess() != AS_private) {
      Ctor->dump();
    };
  }

  // if (!MatchedDecl->getIdentifier() ||
  //     MatchedDecl->getName().startswith("awesome_"))
  //   return;
  // diag(MatchedDecl->getLocation(), "function %0 is insufficiently awesome")
  //     << MatchedDecl
  //     << FixItHint::CreateInsertion(MatchedDecl->getLocation(), "awesome_");
  // diag(MatchedDecl->getLocation(), "insert 'awesome'", DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
