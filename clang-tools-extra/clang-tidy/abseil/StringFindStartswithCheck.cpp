//===--- StringFindStartswithCheck.cc - clang-tidy---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StringFindStartswithCheck.h"

#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"

#include <cassert>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

StringFindStartswithCheck::StringFindStartswithCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringLikeClasses(utils::options::parseStringList(
          Options.get("StringLikeClasses", "::std::basic_string"))),
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.getLocalOrGlobal("IncludeStyle", "llvm"))),
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
          anyOf(hasOperatorName("=="), hasOperatorName("!=")),
          hasEitherOperand(ignoringParenImpCasts(ZeroLiteral)),
          hasEitherOperand(ignoringParenImpCasts(StringFind.bind("findexpr"))))
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

  if (ComparisonExpr->getLocStart().isMacroID())
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
      diag(ComparisonExpr->getLocStart(),
           (StringRef("use ") + StartswithStr + " instead of find() " +
            ComparisonExpr->getOpcodeStr() + " 0")
               .str());

  Diagnostic << FixItHint::CreateReplacement(
      ComparisonExpr->getSourceRange(),
      (StartswithStr + "(" + HaystackExprCode + ", " + NeedleExprCode + ")")
          .str());

  // Create a preprocessor #include FixIt hint (CreateIncludeInsertion checks
  // whether this already exists).
  auto IncludeHint = IncludeInserter->CreateIncludeInsertion(
      Source.getFileID(ComparisonExpr->getLocStart()), AbseilStringsMatchHeader,
      false);
  if (IncludeHint) {
    Diagnostic << *IncludeHint;
  }
}

void StringFindStartswithCheck::registerPPCallbacks(
    CompilerInstance &Compiler) {
  IncludeInserter = llvm::make_unique<clang::tidy::utils::IncludeInserter>(
      Compiler.getSourceManager(), Compiler.getLangOpts(), IncludeStyle);
  Compiler.getPreprocessor().addPPCallbacks(
      IncludeInserter->CreatePPCallbacks());
}

void StringFindStartswithCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClasses));
  Options.store(Opts, "IncludeStyle",
                utils::IncludeSorter::toString(IncludeStyle));
  Options.store(Opts, "AbseilStringsMatchHeader", AbseilStringsMatchHeader);
}

} // namespace abseil
} // namespace tidy
} // namespace clang
