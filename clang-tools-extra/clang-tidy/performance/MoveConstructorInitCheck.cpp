//===--- MoveConstructorInitCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.getLocalOrGlobal("IncludeStyle", "llvm"))) {}

void MoveConstructorInitCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++11; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus11)
    return;

  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isImplicit()), isMoveConstructor(),
          hasAnyConstructorInitializer(
              cxxCtorInitializer(
                  withInitializer(cxxConstructExpr(hasDeclaration(
                      cxxConstructorDecl(isCopyConstructor()).bind("ctor")))))
                  .bind("move-init"))),
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

void MoveConstructorInitCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  Inserter.reset(new utils::IncludeInserter(
      Compiler.getSourceManager(), Compiler.getLangOpts(), IncludeStyle));
  Compiler.getPreprocessor().addPPCallbacks(Inserter->CreatePPCallbacks());
}

void MoveConstructorInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle",
                utils::IncludeSorter::toString(IncludeStyle));
}

} // namespace performance
} // namespace tidy
} // namespace clang
