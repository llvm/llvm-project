//===--- StringFindStartswithCheck.cc - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringFindStartswithCheck.h"

#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

StringFindStartswithCheck::StringFindStartswithCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringLikeClasses(utils::options::parseStringList(
          Options.get("StringLikeClasses", "::std::basic_string"))),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM)),
      AbseilStringsMatchHeader(
          Options.get("AbseilStringsMatchHeader", "absl/strings/match.h")) {}

void StringFindStartswithCheck::registerMatchers(MatchFinder *Finder) {
  auto ZeroLiteral = integerLiteral(equals(0));
  auto StringClassMatcher = cxxRecordDecl(hasAnyName(SmallVector<StringRef, 4>(
      StringLikeClasses.begin(), StringLikeClasses.end())));
  auto StringType = hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(StringClassMatcher)));

  auto StringFind = cxxMemberCallExpr(
      // .find()-call on a string...
      callee(cxxMethodDecl(hasName("find"))),
      on(hasType(StringType)),
      // ... with some search expression ...
      hasArgument(0, expr().bind("needle")),
      // ... and either "0" as second argument or the default argument (also 0).
      anyOf(hasArgument(1, ZeroLiteral), hasArgument(1, cxxDefaultArgExpr())));

  Finder->addMatcher(
      // Match [=!]= with a zero on one side and a string.find on the other.
      binaryOperator(
          hasAnyOperatorName("==", "!="),
          hasOperands(ignoringParenImpCasts(ZeroLiteral),
                      ignoringParenImpCasts(StringFind.bind("findexpr"))))
          .bind("expr"),
      this);
}

void StringFindStartswithCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();

  // Extract matching (sub)expressions
  const auto *ComparisonExpr = Result.Nodes.getNodeAs<BinaryOperator>("expr");
  assert(ComparisonExpr != nullptr);
  const auto *Needle = Result.Nodes.getNodeAs<Expr>("needle");
  assert(Needle != nullptr);
  const Expr *Haystack = Result.Nodes.getNodeAs<CXXMemberCallExpr>("findexpr")
                             ->getImplicitObjectArgument();
  assert(Haystack != nullptr);

  if (ComparisonExpr->getBeginLoc().isMacroID())
    return;

  // Get the source code blocks (as characters) for both the string object
  // and the search expression
  const StringRef NeedleExprCode = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Needle->getSourceRange()), Source,
      Context.getLangOpts());
  const StringRef HaystackExprCode = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Haystack->getSourceRange()), Source,
      Context.getLangOpts());

  // Create the StartsWith string, negating if comparison was "!=".
  bool Neg = ComparisonExpr->getOpcodeStr() == "!=";
  StringRef StartswithStr;
  if (Neg) {
    StartswithStr = "!absl::StartsWith";
  } else {
    StartswithStr = "absl::StartsWith";
  }

  // Create the warning message and a FixIt hint replacing the original expr.
  auto Diagnostic =
      diag(ComparisonExpr->getBeginLoc(),
           (StringRef("use ") + StartswithStr + " instead of find() " +
            ComparisonExpr->getOpcodeStr() + " 0")
               .str());

  Diagnostic << FixItHint::CreateReplacement(
      ComparisonExpr->getSourceRange(),
      (StartswithStr + "(" + HaystackExprCode + ", " + NeedleExprCode + ")")
          .str());

  // Create a preprocessor #include FixIt hint (CreateIncludeInsertion checks
  // whether this already exists).
  Diagnostic << IncludeInserter.createIncludeInsertion(
      Source.getFileID(ComparisonExpr->getBeginLoc()),
      AbseilStringsMatchHeader);
}

void StringFindStartswithCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void StringFindStartswithCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClasses));
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  Options.store(Opts, "AbseilStringsMatchHeader", AbseilStringsMatchHeader);
}

} // namespace abseil
} // namespace tidy
} // namespace clang
