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
namespace misc {

namespace {

unsigned int
parmVarDeclRefExprOccurences(const ParmVarDecl &MovableParam,
                             const CXXConstructorDecl &ConstructorDecl,
                             ASTContext &Context) {
  unsigned int Occurrences = 0;
  auto AllDeclRefs =
      findAll(declRefExpr(to(parmVarDecl(equalsNode(&MovableParam)))));
  Occurrences += match(AllDeclRefs, *ConstructorDecl.getBody(), Context).size();
  for (const auto *Initializer : ConstructorDecl.inits()) {
    Occurrences += match(AllDeclRefs, *Initializer->getInit(), Context).size();
  }
  return Occurrences;
}

} // namespace

MoveConstructorInitCheck::MoveConstructorInitCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.get("IncludeStyle", "llvm"))),
      UseCERTSemantics(Options.get("UseCERTSemantics", 0) != 0) {}

void MoveConstructorInitCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++11; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus11)
    return;

  Finder->addMatcher(
      cxxConstructorDecl(
          unless(isImplicit()),
          allOf(isMoveConstructor(),
                hasAnyConstructorInitializer(
                    cxxCtorInitializer(
                        withInitializer(cxxConstructExpr(hasDeclaration(
                            cxxConstructorDecl(isCopyConstructor())
                                .bind("ctor")))))
                        .bind("move-init")))),
      this);

  auto NonConstValueMovableAndExpensiveToCopy =
      qualType(allOf(unless(pointerType()), unless(isConstQualified()),
                     hasDeclaration(cxxRecordDecl(hasMethod(cxxConstructorDecl(
                         isMoveConstructor(), unless(isDeleted()))))),
                     matchers::isExpensiveToCopy()));

  // This checker is also used to implement cert-oop11-cpp, but when using that
  // form of the checker, we do not want to diagnose movable parameters.
  if (!UseCERTSemantics) {
    Finder->addMatcher(
        cxxConstructorDecl(
            allOf(
                unless(isMoveConstructor()),
                hasAnyConstructorInitializer(withInitializer(cxxConstructExpr(
                    hasDeclaration(cxxConstructorDecl(isCopyConstructor())),
                    hasArgument(
                        0,
                        declRefExpr(
                            to(parmVarDecl(
                                   hasType(
                                       NonConstValueMovableAndExpensiveToCopy))
                                   .bind("movable-param")))
                            .bind("init-arg")))))))
            .bind("ctor-decl"),
        this);
  }
}

void MoveConstructorInitCheck::check(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<CXXCtorInitializer>("move-init") != nullptr)
    handleMoveConstructor(Result);
  if (Result.Nodes.getNodeAs<ParmVarDecl>("movable-param") != nullptr)
    handleParamNotMoved(Result);
}

void MoveConstructorInitCheck::handleParamNotMoved(
    const MatchFinder::MatchResult &Result) {
  const auto *MovableParam =
      Result.Nodes.getNodeAs<ParmVarDecl>("movable-param");
  const auto *ConstructorDecl =
      Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor-decl");
  const auto *InitArg = Result.Nodes.getNodeAs<DeclRefExpr>("init-arg");
  // If the parameter is referenced more than once it is not safe to move it.
  if (parmVarDeclRefExprOccurences(*MovableParam, *ConstructorDecl,
                                   *Result.Context) > 1)
    return;
  auto DiagOut = diag(InitArg->getLocStart(),
                      "value argument %0 can be moved to avoid copy")
                 << MovableParam;
  DiagOut << FixItHint::CreateReplacement(
      InitArg->getSourceRange(),
      (Twine("std::move(") + MovableParam->getName() + ")").str());
  if (auto IncludeFixit = Inserter->CreateIncludeInsertion(
          Result.SourceManager->getFileID(InitArg->getLocStart()), "utility",
          /*IsAngled=*/true)) {
    DiagOut << *IncludeFixit;
  }
}

void MoveConstructorInitCheck::handleMoveConstructor(
    const MatchFinder::MatchResult &Result) {
  const auto *CopyCtor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  const auto *Initializer = Result.Nodes.getNodeAs<CXXCtorInitializer>("move-init");

  // Do not diagnose if the expression used to perform the initialization is a
  // trivially-copyable type.
  QualType QT = Initializer->getInit()->getType();
  if (QT.isTriviallyCopyableType(*Result.Context))
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
  Options.store(Opts, "UseCERTSemantics", UseCERTSemantics ? 1 : 0);
}

} // namespace misc
} // namespace tidy
} // namespace clang
