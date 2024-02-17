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
#include "clang/Lex/Lexer.h"

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

bool hasPrivateConstructor(const CXXRecordDecl *RD) {
  for (auto &&Ctor : RD->ctors()) {
    if (Ctor->getAccess() == AS_private)
      return true;
  }

  return false;
}

bool isDerivedBefriended(const CXXRecordDecl *CRTP,
                         const CXXRecordDecl *Derived) {
  for (auto &&Friend : CRTP->friends()) {
    if (Friend->getFriendType()->getType()->getAsCXXRecordDecl() == Derived)
      return true;
  }

  return false;
}
} // namespace

void UnsafeCrtpCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      classTemplateSpecializationDecl(
          decl().bind("crtp"),
          hasAnyTemplateArgument(refersToType(recordType(hasDeclaration(
              cxxRecordDecl(isDerivedFrom(cxxRecordDecl(isBoundNode("crtp"))))
                  .bind("derived")))))),
      this);
}

void UnsafeCrtpCheck::check(const MatchFinder::MatchResult &Result) {

  const auto *MatchedCRTP =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("crtp");
  const auto *MatchedDerived = Result.Nodes.getNodeAs<CXXRecordDecl>("derived");

  if (!MatchedCRTP->hasUserDeclaredConstructor()) {
    diag(MatchedCRTP->getLocation(),
         "the implicit default constructor of the CRTP is publicly accessible")
        << MatchedCRTP
        << FixItHint::CreateInsertion(
               MatchedCRTP->getBraceRange().getBegin().getLocWithOffset(1),
               "private: " + MatchedCRTP->getNameAsString() + "() = default;");

    diag(MatchedCRTP->getLocation(), "consider making it private",
         DiagnosticIDs::Note);
  }

  // FIXME: Extract this.
  size_t idx = 0;
  for (auto &&TemplateArg : MatchedCRTP->getTemplateArgs().asArray()) {
    if (TemplateArg.getKind() == TemplateArgument::Type &&
        TemplateArg.getAsType()->getAsCXXRecordDecl() == MatchedDerived) {
      break;
    }
    ++idx;
  }

  if (hasPrivateConstructor(MatchedCRTP) &&
      !isDerivedBefriended(MatchedCRTP, MatchedDerived)) {
    diag(MatchedCRTP->getLocation(),
         "the CRTP cannot be constructed from the derived class")
        << MatchedCRTP
        << FixItHint::CreateInsertion(
               MatchedCRTP->getBraceRange().getEnd().getLocWithOffset(-1),
               "friend " +
                   MatchedCRTP->getSpecializedTemplate()
                       ->getTemplateParameters()
                       ->asArray()[idx]
                       ->getNameAsString() +
                   ';');
    diag(MatchedCRTP->getLocation(),
         "consider declaring the derived class as friend", DiagnosticIDs::Note);
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
