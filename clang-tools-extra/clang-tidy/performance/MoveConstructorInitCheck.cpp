//===--- MoveConstructorInitCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveConstructorInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

MoveConstructorInitCheck::MoveConstructorInitCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void MoveConstructorInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxConstructorDecl(
                   unless(isImplicit()), isMoveConstructor(),
                   hasAnyConstructorInitializer(
                       cxxCtorInitializer(
                           withInitializer(cxxConstructExpr(hasDeclaration(
                               cxxConstructorDecl(isCopyConstructor())
                                   .bind("ctor")))))
                           .bind("move-init")))),
      this);
}

void MoveConstructorInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CopyCtor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  const auto *Initializer =
      Result.Nodes.getNodeAs<CXXCtorInitializer>("move-init");

  // Do not diagnose if the expression used to perform the initialization is a
  // trivially-copyable type.
  QualType QT = Initializer->getInit()->getType();
  if (QT.isTriviallyCopyableType(*Result.Context))
    return;

  if (QT.isConstQualified())
    return;

  const auto *RD = QT->getAsCXXRecordDecl();
  if (RD && RD->isTriviallyCopyable())
    return;

  // Diagnose when the class type has a move constructor available, but the
  // ctor-initializer uses the copy constructor instead.
  const CXXConstructorDecl *Candidate = nullptr;
  for (const auto *Ctor : CopyCtor->getParent()->ctors()) {
    if (Ctor->isMoveConstructor() && Ctor->getAccess() <= AS_protected &&
        !Ctor->isDeleted()) {
      // The type has a move constructor that is at least accessible to the
      // initializer.
      //
      // FIXME: Determine whether the move constructor is a viable candidate
      // for the ctor-initializer, perhaps provide a fix-it that suggests
      // using std::move().
      Candidate = Ctor;
      break;
    }
  }

  if (Candidate) {
    // There's a move constructor candidate that the caller probably intended
    // to call instead.
    diag(Initializer->getSourceLocation(),
         "move constructor initializes %select{class member|base class}0 by "
         "calling a copy constructor")
        << Initializer->isBaseInitializer();
    diag(CopyCtor->getLocation(), "copy constructor being called",
         DiagnosticIDs::Note);
    diag(Candidate->getLocation(), "candidate move constructor here",
         DiagnosticIDs::Note);
  }
}

} // namespace clang::tidy::performance
