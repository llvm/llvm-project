//===--- UnsafeCrtpCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeCrtpCheck.h"
#include "../utils/LexerUtils.h"
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

std::vector<FixItHint> hintMakeCtorPrivate(const CXXConstructorDecl *Ctor,
                                           std::string OriginalAccess,
                                           const SourceManager &SM,
                                           const LangOptions &LangOpts) {
  std::vector<FixItHint> Hints;

  Hints.emplace_back(FixItHint::CreateInsertion(
      Ctor->getBeginLoc().getLocWithOffset(-1), "private:\n"));

  Hints.emplace_back(FixItHint::CreateInsertion(
      Ctor->isExplicitlyDefaulted()
          ? utils::lexer::findNextTerminator(Ctor->getEndLoc(), SM, LangOpts)
                .getLocWithOffset(1)
          : Ctor->getEndLoc().getLocWithOffset(1),
      '\n' + OriginalAccess + ':' + '\n'));

  return Hints;
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
  const auto *CRTPInstantiation =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("crtp");
  const auto *Derived = Result.Nodes.getNodeAs<CXXRecordDecl>("derived");

  const CXXRecordDecl *CRTPDeclaration =
      CRTPInstantiation->getSpecializedTemplate()->getTemplatedDecl();

  if (!CRTPDeclaration->hasUserDeclaredConstructor()) {
    diag(CRTPDeclaration->getLocation(),
         "the implicit default constructor of the CRTP is publicly accessible")
        << CRTPDeclaration
        << FixItHint::CreateInsertion(
               CRTPDeclaration->getBraceRange().getBegin().getLocWithOffset(1),
               (CRTPDeclaration->isStruct() ? "\nprivate:\n" : "\n") +
                   CRTPDeclaration->getNameAsString() + "() = default;" +
                   (CRTPDeclaration->isStruct() ? "\npublic:\n" : "\n"));
    diag(CRTPDeclaration->getLocation(), "consider making it private",
         DiagnosticIDs::Note);
  }

  const auto *DerivedTemplateParameter =
      *getDerivedParameter(CRTPInstantiation, Derived);

  if (hasPrivateConstructor(CRTPDeclaration) &&
      !isDerivedParameterBefriended(CRTPDeclaration,
                                    DerivedTemplateParameter)) {
    diag(CRTPDeclaration->getLocation(),
         "the CRTP cannot be constructed from the derived class")
        << CRTPDeclaration
        << FixItHint::CreateInsertion(
               CRTPDeclaration->getBraceRange().getEnd(),
               "friend " + DerivedTemplateParameter->getNameAsString() + ';' +
                   '\n');
    diag(CRTPDeclaration->getLocation(),
         "consider declaring the derived class as friend", DiagnosticIDs::Note);
  }

  for (auto &&Ctor : CRTPDeclaration->ctors()) {
    if (Ctor->getAccess() == AS_private)
      continue;

    bool IsPublic = Ctor->getAccess() == AS_public;
    std::string Access = IsPublic ? "public" : "protected";

    diag(Ctor->getLocation(),
         "%0 contructor allows the CRTP to be %select{inherited "
         "from|constructed}1 as a regular template class")
        << Access << IsPublic << Ctor
        << hintMakeCtorPrivate(Ctor, Access, *Result.SourceManager,
                               getLangOpts());
    diag(Ctor->getLocation(), "consider making it private",
         DiagnosticIDs::Note);
  }
}

} // namespace clang::tidy::bugprone
