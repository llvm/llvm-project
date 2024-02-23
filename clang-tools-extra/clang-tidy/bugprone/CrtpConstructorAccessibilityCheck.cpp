//===--- CrtpConstructorAccessibilityCheck.cpp - clang-tidy
//---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrtpConstructorAccessibilityCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static bool hasPrivateConstructor(const CXXRecordDecl *RD) {
  return llvm::any_of(RD->ctors(), [](const CXXConstructorDecl *Ctor) {
    return Ctor->getAccess() == AS_private;
  });
}

static bool isDerivedParameterBefriended(const CXXRecordDecl *CRTP,
                                         const NamedDecl *Param) {
  return llvm::any_of(CRTP->friends(), [&](const FriendDecl *Friend) {
    const auto *TTPT =
        dyn_cast<TemplateTypeParmType>(Friend->getFriendType()->getType());

    return TTPT && TTPT->getDecl() == Param;
  });
}

static bool isDerivedClassBefriended(const CXXRecordDecl *CRTP,
                                     const CXXRecordDecl *Derived) {
  return llvm::any_of(CRTP->friends(), [&](const FriendDecl *Friend) {
    return Friend->getFriendType()->getType()->getAsCXXRecordDecl() == Derived;
  });
}

static const NamedDecl *
getDerivedParameter(const ClassTemplateSpecializationDecl *CRTP,
                    const CXXRecordDecl *Derived) {
  size_t Idx = 0;
  const bool AnyOf = llvm::any_of(
      CRTP->getTemplateArgs().asArray(), [&](const TemplateArgument &Arg) {
        ++Idx;
        return Arg.getKind() == TemplateArgument::Type &&
               Arg.getAsType()->getAsCXXRecordDecl() == Derived;
      });

  return AnyOf ? CRTP->getSpecializedTemplate()
                     ->getTemplateParameters()
                     ->getParam(Idx - 1)
               : nullptr;
}

static std::vector<FixItHint>
hintMakeCtorPrivate(const CXXConstructorDecl *Ctor,
                    const std::string &OriginalAccess) {
  std::vector<FixItHint> Hints;

  Hints.emplace_back(FixItHint::CreateInsertion(
      Ctor->getBeginLoc().getLocWithOffset(-1), "private:\n"));

  const ASTContext &ASTCtx = Ctor->getASTContext();
  const SourceLocation CtorEndLoc =
      Ctor->isExplicitlyDefaulted()
          ? utils::lexer::findNextTerminator(Ctor->getEndLoc(),
                                             ASTCtx.getSourceManager(),
                                             ASTCtx.getLangOpts())
          : Ctor->getEndLoc();
  Hints.emplace_back(FixItHint::CreateInsertion(
      CtorEndLoc.getLocWithOffset(1), '\n' + OriginalAccess + ':' + '\n'));

  return Hints;
}

void CrtpConstructorAccessibilityCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      classTemplateSpecializationDecl(
          decl().bind("crtp"),
          hasAnyTemplateArgument(refersToType(recordType(hasDeclaration(
              cxxRecordDecl(
                  isDerivedFrom(cxxRecordDecl(equalsBoundNode("crtp"))))
                  .bind("derived")))))),
      this);
}

void CrtpConstructorAccessibilityCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *CRTPInstantiation =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("crtp");
  const auto *DerivedRecord = Result.Nodes.getNodeAs<CXXRecordDecl>("derived");
  const CXXRecordDecl *CRTPDeclaration =
      CRTPInstantiation->getSpecializedTemplate()->getTemplatedDecl();

  const auto *DerivedTemplateParameter =
      getDerivedParameter(CRTPInstantiation, DerivedRecord);

  assert(DerivedTemplateParameter &&
         "No template parameter corresponds to the derived class of the CRTP.");

  bool NeedsFriend = !isDerivedParameterBefriended(CRTPDeclaration,
                                                   DerivedTemplateParameter) &&
                     !isDerivedClassBefriended(CRTPDeclaration, DerivedRecord);

  const FixItHint HintFriend = FixItHint::CreateInsertion(
      CRTPDeclaration->getBraceRange().getEnd(),
      "friend " + DerivedTemplateParameter->getNameAsString() + ';' + '\n');

  if (!CRTPDeclaration->hasUserDeclaredConstructor()) {
    const bool IsStruct = CRTPDeclaration->isStruct();

    // FIXME: Clean this up!
    {
      DiagnosticBuilder Diag =
          diag(CRTPDeclaration->getLocation(),
               "the implicit default constructor of the CRTP is publicly "
               "accessible")
          << CRTPDeclaration
          << FixItHint::CreateInsertion(
                 CRTPDeclaration->getBraceRange().getBegin().getLocWithOffset(
                     1),
                 (IsStruct ? "\nprivate:\n" : "\n") +
                     CRTPDeclaration->getNameAsString() + "() = default;\n" +
                     (IsStruct ? "public:\n" : ""));

      if (NeedsFriend)
        Diag << HintFriend;
    }

    diag(CRTPDeclaration->getLocation(),
         "consider making it private%select{| and declaring the derived class "
         "as friend}0",
         DiagnosticIDs::Note)
        << NeedsFriend;
  }

  if (hasPrivateConstructor(CRTPDeclaration) && NeedsFriend) {
    diag(CRTPDeclaration->getLocation(),
         "the CRTP cannot be constructed from the derived class")
        << CRTPDeclaration << HintFriend;
    diag(CRTPDeclaration->getLocation(),
         "consider declaring the derived class as friend", DiagnosticIDs::Note);
  }

  for (auto &&Ctor : CRTPDeclaration->ctors()) {
    if (Ctor->getAccess() == AS_private)
      continue;

    const bool IsPublic = Ctor->getAccess() == AS_public;
    const std::string Access = IsPublic ? "public" : "protected";

    // FIXME: Clean this up!
    {
      DiagnosticBuilder Diag =
          diag(Ctor->getLocation(),
               "%0 contructor allows the CRTP to be %select{inherited "
               "from|constructed}1 as a regular template class")
          << Access << IsPublic << Ctor << hintMakeCtorPrivate(Ctor, Access);

      if (NeedsFriend)
        Diag << HintFriend;
    }

    // FIXME: Test the NeedsFriend false case!
    diag(Ctor->getLocation(),
         "consider making it private%select{| and declaring the derived class "
         "as friend}0",
         DiagnosticIDs::Note)
        << NeedsFriend;
  }
}

bool CrtpConstructorAccessibilityCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus11;
}
} // namespace clang::tidy::bugprone
