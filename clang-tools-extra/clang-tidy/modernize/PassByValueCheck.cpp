//===--- PassByValueCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassByValueCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;
using namespace llvm;

namespace clang {
namespace tidy {
namespace modernize {

namespace {
/// Matches move-constructible classes.
///
/// Given
/// \code
///   // POD types are trivially move constructible.
///   struct Foo { int a; };
///
///   struct Bar {
///     Bar(Bar &&) = deleted;
///     int a;
///   };
/// \endcode
/// recordDecl(isMoveConstructible())
///   matches "Foo".
AST_MATCHER(CXXRecordDecl, isMoveConstructible) {
  for (const CXXConstructorDecl *Ctor : Node.ctors()) {
    if (Ctor->isMoveConstructor() && !Ctor->isDeleted())
      return true;
  }
  return false;
}
} // namespace

static TypeMatcher notTemplateSpecConstRefType() {
  return lValueReferenceType(
      pointee(unless(templateSpecializationType()), isConstQualified()));
}

static TypeMatcher nonConstValueType() {
  return qualType(unless(anyOf(referenceType(), isConstQualified())));
}

/// Whether or not \p ParamDecl is used exactly one time in \p Ctor.
///
/// Checks both in the init-list and the body of the constructor.
static bool paramReferredExactlyOnce(const CXXConstructorDecl *Ctor,
                                     const ParmVarDecl *ParamDecl) {
  /// \c clang::RecursiveASTVisitor that checks that the given
  /// \c ParmVarDecl is used exactly one time.
  ///
  /// \see ExactlyOneUsageVisitor::hasExactlyOneUsageIn()
  class ExactlyOneUsageVisitor
      : public RecursiveASTVisitor<ExactlyOneUsageVisitor> {
    friend class RecursiveASTVisitor<ExactlyOneUsageVisitor>;

  public:
    ExactlyOneUsageVisitor(const ParmVarDecl *ParamDecl)
        : ParamDecl(ParamDecl) {}

    /// Whether or not the parameter variable is referred only once in
    /// the
    /// given constructor.
    bool hasExactlyOneUsageIn(const CXXConstructorDecl *Ctor) {
      Count = 0;
      TraverseDecl(const_cast<CXXConstructorDecl *>(Ctor));
      return Count == 1;
    }

  private:
    /// Counts the number of references to a variable.
    ///
    /// Stops the AST traversal if more than one usage is found.
    bool VisitDeclRefExpr(DeclRefExpr *D) {
      if (const ParmVarDecl *To = dyn_cast<ParmVarDecl>(D->getDecl())) {
        if (To == ParamDecl) {
          ++Count;
          if (Count > 1) {
            // No need to look further, used more than once.
            return false;
          }
        }
      }
      return true;
    }

    const ParmVarDecl *ParamDecl;
    unsigned Count;
  };

  return ExactlyOneUsageVisitor(ParamDecl).hasExactlyOneUsageIn(Ctor);
}

/// Find all references to \p ParamDecl across all of the
/// redeclarations of \p Ctor.
static SmallVector<const ParmVarDecl *, 2>
collectParamDecls(const CXXConstructorDecl *Ctor,
                  const ParmVarDecl *ParamDecl) {
  SmallVector<const ParmVarDecl *, 2> Results;
  unsigned ParamIdx = ParamDecl->getFunctionScopeIndex();

  for (const FunctionDecl *Redecl : Ctor->redecls())
    Results.push_back(Redecl->getParamDecl(ParamIdx));
  return Results;
}

PassByValueCheck::PassByValueCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM)),
      ValuesOnly(Options.get("ValuesOnly", false)) {}

void PassByValueCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "ValuesOnly", ValuesOnly);
}

void PassByValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxConstructorDecl(
              forEachConstructorInitializer(
                  cxxCtorInitializer(
                      unless(isBaseInitializer()),
                      // Clang builds a CXXConstructExpr only when it knows
                      // which constructor will be called. In dependent contexts
                      // a ParenListExpr is generated instead of a
                      // CXXConstructExpr, filtering out templates automatically
                      // for us.
                      withInitializer(cxxConstructExpr(
                          has(ignoringParenImpCasts(declRefExpr(to(
                              parmVarDecl(
                                  hasType(qualType(
                                      // Match only const-ref or a non-const
                                      // value parameters. Rvalues,
                                      // TemplateSpecializationValues and
                                      // const-values shouldn't be modified.
                                      ValuesOnly
                                          ? nonConstValueType()
                                          : anyOf(notTemplateSpecConstRefType(),
                                                  nonConstValueType()))))
                                  .bind("Param"))))),
                          hasDeclaration(cxxConstructorDecl(
                              isCopyConstructor(), unless(isDeleted()),
                              hasDeclContext(
                                  cxxRecordDecl(isMoveConstructible())))))))
                      .bind("Initializer")))
              .bind("Ctor")),
      this);
}

void PassByValueCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void PassByValueCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("Ctor");
  const auto *ParamDecl = Result.Nodes.getNodeAs<ParmVarDecl>("Param");
  const auto *Initializer =
      Result.Nodes.getNodeAs<CXXCtorInitializer>("Initializer");
  SourceManager &SM = *Result.SourceManager;

  // If the parameter is used or anything other than the copy, do not apply
  // the changes.
  if (!paramReferredExactlyOnce(Ctor, ParamDecl))
    return;

  // If the parameter is trivial to copy, don't move it. Moving a trivivally
  // copyable type will cause a problem with performance-move-const-arg
  if (ParamDecl->getType().getNonReferenceType().isTriviallyCopyableType(
          *Result.Context))
    return;

  auto Diag = diag(ParamDecl->getBeginLoc(), "pass by value and use std::move");

  // Iterate over all declarations of the constructor.
  for (const ParmVarDecl *ParmDecl : collectParamDecls(Ctor, ParamDecl)) {
    auto ParamTL = ParmDecl->getTypeSourceInfo()->getTypeLoc();
    auto RefTL = ParamTL.getAs<ReferenceTypeLoc>();

    // Do not replace if it is already a value, skip.
    if (RefTL.isNull())
      continue;

    TypeLoc ValueTL = RefTL.getPointeeLoc();
    auto TypeRange = CharSourceRange::getTokenRange(ParmDecl->getBeginLoc(),
                                                    ParamTL.getEndLoc());
    std::string ValueStr = Lexer::getSourceText(CharSourceRange::getTokenRange(
                                                    ValueTL.getSourceRange()),
                                                SM, getLangOpts())
                               .str();
    ValueStr += ' ';
    Diag << FixItHint::CreateReplacement(TypeRange, ValueStr);
  }

  // Use std::move in the initialization list.
  Diag << FixItHint::CreateInsertion(Initializer->getRParenLoc(), ")")
       << FixItHint::CreateInsertion(
              Initializer->getLParenLoc().getLocWithOffset(1), "std::move(")
       << Inserter.createIncludeInsertion(
              Result.SourceManager->getFileID(Initializer->getSourceLocation()),
              "<utility>");
}

} // namespace modernize
} // namespace tidy
} // namespace clang
