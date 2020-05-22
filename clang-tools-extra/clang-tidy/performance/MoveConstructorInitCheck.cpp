//===--- MoveConstructorInitCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveConstructorInitCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

MoveConstructorInitCheck::MoveConstructorInitCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(Options.getLocalOrGlobal("IncludeStyle",
                                            utils::IncludeSorter::getMapping(),
                                            utils::IncludeSorter::IS_LLVM)) {}

void MoveConstructorInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
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
      // for the ctor-initializer, perhaps provide a fixit that suggests
      // using std::move().
      Candidate = Ctor;
      break;
    }
  }

  if (Candidate) {
    // There's a move constructor candidate that the caller probably intended
    // to call instead.
    diag(Initializer->getSourceLocation(),
         "move constructor initializes %0 by calling a copy constructor")
        << (Initializer->isBaseInitializer() ? "base class" : "class member");
    diag(CopyCtor->getLocation(), "copy constructor being called",
         DiagnosticIDs::Note);
    diag(Candidate->getLocation(), "candidate move constructor here",
         DiagnosticIDs::Note);
  }
}

void MoveConstructorInitCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter = std::make_unique<utils::IncludeInserter>(SM, getLangOpts(),
                                                       IncludeStyle);
  PP->addPPCallbacks(Inserter->CreatePPCallbacks());
}

void MoveConstructorInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeStyle,
                utils::IncludeSorter::getMapping());
}

} // namespace performance
} // namespace tidy
} // namespace clang
