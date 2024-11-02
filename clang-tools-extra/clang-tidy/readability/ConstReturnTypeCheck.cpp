//===--- ConstReturnTypeCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstReturnTypeCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/Optional.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

// Finds the location of the qualifying `const` token in the `FunctionDecl`'s
// return type. Returns `None` when the return type is not `const`-qualified or
// `const` does not appear in `Def`'s source, like when the type is an alias or
// a macro.
static llvm::Optional<Token>
findConstToRemove(const FunctionDecl *Def,
                  const MatchFinder::MatchResult &Result) {
  if (!Def->getReturnType().isLocalConstQualified())
    return std::nullopt;

  // Get the begin location for the function name, including any qualifiers
  // written in the source (for out-of-line declarations). A FunctionDecl's
  // "location" is the start of its name, so, when the name is unqualified, we
  // use `getLocation()`.
  SourceLocation NameBeginLoc = Def->getQualifier()
                                    ? Def->getQualifierLoc().getBeginLoc()
                                    : Def->getLocation();
  // Since either of the locs can be in a macro, use `makeFileCharRange` to be
  // sure that we have a consistent `CharSourceRange`, located entirely in the
  // source file.
  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Def->getBeginLoc(), NameBeginLoc),
      *Result.SourceManager, Result.Context->getLangOpts());

  if (FileRange.isInvalid())
    return std::nullopt;

  return utils::lexer::getQualifyingToken(
      tok::kw_const, FileRange, *Result.Context, *Result.SourceManager);
}

namespace {

AST_MATCHER(QualType, isLocalConstQualified) {
  return Node.isLocalConstQualified();
}

AST_MATCHER(QualType, isTypeOfType) {
  return isa<TypeOfType>(Node.getTypePtr());
}

AST_MATCHER(QualType, isTypeOfExprType) {
  return isa<TypeOfExprType>(Node.getTypePtr());
}

struct CheckResult {
  // Source range of the relevant `const` token in the definition being checked.
  CharSourceRange ConstRange;

  // FixItHints associated with the definition being checked.
  llvm::SmallVector<clang::FixItHint, 4> Hints;

  // Locations of any declarations that could not be fixed.
  llvm::SmallVector<clang::SourceLocation, 4> DeclLocs;
};

} // namespace

// Does the actual work of the check.
static CheckResult checkDef(const clang::FunctionDecl *Def,
                            const MatchFinder::MatchResult &MatchResult) {
  CheckResult Result;
  llvm::Optional<Token> Tok = findConstToRemove(Def, MatchResult);
  if (!Tok)
    return Result;

  Result.ConstRange =
      CharSourceRange::getCharRange(Tok->getLocation(), Tok->getEndLoc());
  Result.Hints.push_back(FixItHint::CreateRemoval(Result.ConstRange));

  // Fix the definition and any visible declarations, but don't warn
  // separately for each declaration. Instead, associate all fixes with the
  // single warning at the definition.
  for (const FunctionDecl *Decl = Def->getPreviousDecl(); Decl != nullptr;
       Decl = Decl->getPreviousDecl()) {
    if (llvm::Optional<Token> T = findConstToRemove(Decl, MatchResult))
      Result.Hints.push_back(FixItHint::CreateRemoval(
          CharSourceRange::getCharRange(T->getLocation(), T->getEndLoc())));
    else
      // `getInnerLocStart` gives the start of the return type.
      Result.DeclLocs.push_back(Decl->getInnerLocStart());
  }
  return Result;
}

void ConstReturnTypeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void ConstReturnTypeCheck::registerMatchers(MatchFinder *Finder) {
  // Find all function definitions for which the return types are `const`
  // qualified, ignoring decltype types.
  auto NonLocalConstType = qualType(
      unless(isLocalConstQualified()),
      anyOf(decltypeType(), autoType(), isTypeOfType(), isTypeOfExprType()));
  Finder->addMatcher(
      functionDecl(
          returns(allOf(isConstQualified(), unless(NonLocalConstType))),
          anyOf(isDefinition(), cxxMethodDecl(isPure())),
          // Overridden functions are not actionable.
          unless(cxxMethodDecl(isOverride())))
          .bind("func"),
      this);
}

void ConstReturnTypeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Def = Result.Nodes.getNodeAs<FunctionDecl>("func");
  // Suppress the check if macros are involved.
  if (IgnoreMacros &&
      (Def->getBeginLoc().isMacroID() || Def->getEndLoc().isMacroID()))
    return;

  CheckResult CR = checkDef(Def, Result);
  {
    // Clang only supports one in-flight diagnostic at a time. So, delimit the
    // scope of `Diagnostic` to allow further diagnostics after the scope.  We
    // use `getInnerLocStart` to get the start of the return type.
    DiagnosticBuilder Diagnostic =
        diag(Def->getInnerLocStart(),
             "return type %0 is 'const'-qualified at the top level, which may "
             "reduce code readability without improving const correctness")
        << Def->getReturnType();
    if (CR.ConstRange.isValid())
      Diagnostic << CR.ConstRange;

    // Do not propose fixes for virtual function.
    const auto *Method = dyn_cast<CXXMethodDecl>(Def);
    if (Method && Method->isVirtual())
      return;

    for (auto &Hint : CR.Hints)
      Diagnostic << Hint;
  }
  for (auto Loc : CR.DeclLocs)
    diag(Loc, "could not transform this declaration", DiagnosticIDs::Note);
}

} // namespace readability
} // namespace tidy
} // namespace clang
