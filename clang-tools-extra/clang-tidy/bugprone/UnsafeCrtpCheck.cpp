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

bool isDerivedParameterBefriended(const CXXRecordDecl *CRTP,
                                  const NamedDecl *Derived) {
  for (auto &&Friend : CRTP->friends()) {
    const auto *TTPT =
        dyn_cast<TemplateTypeParmType>(Friend->getFriendType()->getType());

    if (TTPT && TTPT->getDecl() == Derived)
      return true;
  }

  return false;
}

std::optional<const NamedDecl *>
getDerivedParameter(const ClassTemplateSpecializationDecl *CRTP,
                    const CXXRecordDecl *Derived) {
  size_t Idx = 0;
  bool Found = false;
  for (auto &&TemplateArg : CRTP->getTemplateArgs().asArray()) {
    if (TemplateArg.getKind() == TemplateArgument::Type &&
        TemplateArg.getAsType()->getAsCXXRecordDecl() == Derived) {
      Found = true;
      break;
    }
    ++Idx;
  }

  if (!Found)
    return std::nullopt;

  return CRTP->getSpecializedTemplate()->getTemplateParameters()->getParam(Idx);
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

  const auto *CRTPTemplate =
      MatchedCRTP->getSpecializedTemplate()->getTemplatedDecl();

  if (!CRTPTemplate->hasUserDeclaredConstructor()) {
    diag(CRTPTemplate->getLocation(),
         "the implicit default constructor of the CRTP is publicly accessible")
        << CRTPTemplate
        << FixItHint::CreateInsertion(
               CRTPTemplate->getBraceRange().getBegin().getLocWithOffset(1),
               "private: " + CRTPTemplate->getNameAsString() + "() = default;");

    diag(CRTPTemplate->getLocation(), "consider making it private",
         DiagnosticIDs::Note);
  }

  const auto *DerivedTemplateParameter =
      *getDerivedParameter(MatchedCRTP, MatchedDerived);

  if (hasPrivateConstructor(CRTPTemplate) &&
      !isDerivedParameterBefriended(CRTPTemplate, DerivedTemplateParameter)) {
    diag(CRTPTemplate->getLocation(),
         "the CRTP cannot be constructed from the derived class")
        << CRTPTemplate
        << FixItHint::CreateInsertion(
               CRTPTemplate->getBraceRange().getEnd().getLocWithOffset(-1),
               "friend " + DerivedTemplateParameter->getNameAsString() + ';');
    diag(CRTPTemplate->getLocation(),
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
