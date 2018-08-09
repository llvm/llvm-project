//===--- SuspiciousMissingCommaCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SuspiciousMissingCommaCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {

bool isConcatenatedLiteralsOnPurpose(ASTContext *Ctx,
                                     const StringLiteral *Lit) {
  // String literals surrounded by parentheses are assumed to be on purpose.
  //    i.e.:  const char* Array[] = { ("a" "b" "c"), "d", [...] };
  auto Parents = Ctx->getParents(*Lit);
  if (Parents.size() == 1 && Parents[0].get<ParenExpr>() != nullptr)
    return true;

  // Appropriately indented string literals are assumed to be on purpose.
  // The following frequent indentation is accepted:
  //     const char* Array[] = {
  //       "first literal"
  //           "indented literal"
  //           "indented literal",
  //       "second literal",
  //       [...]
  //     };
  const SourceManager &SM = Ctx->getSourceManager();
  bool IndentedCorrectly = true;
  SourceLocation FirstToken = Lit->getStrTokenLoc(0);
  FileID BaseFID = SM.getFileID(FirstToken);
  unsigned int BaseIndent = SM.getSpellingColumnNumber(FirstToken);
  unsigned int BaseLine = SM.getSpellingLineNumber(FirstToken);
  for (unsigned int TokNum = 1; TokNum < Lit->getNumConcatenated(); ++TokNum) {
    SourceLocation Token = Lit->getStrTokenLoc(TokNum);
    FileID FID = SM.getFileID(Token);
    unsigned int Indent = SM.getSpellingColumnNumber(Token);
    unsigned int Line = SM.getSpellingLineNumber(Token);
    if (FID != BaseFID || Line != BaseLine + TokNum || Indent <= BaseIndent) {
      IndentedCorrectly = false;
      break;
    }
  }
  if (IndentedCorrectly)
    return true;

  // There is no pattern recognized by the checker, assume it's not on purpose.
  return false;
}

AST_MATCHER_P(StringLiteral, isConcatenatedLiteral, unsigned,
              MaxConcatenatedTokens) {
  return Node.getNumConcatenated() > 1 &&
         Node.getNumConcatenated() < MaxConcatenatedTokens &&
         !isConcatenatedLiteralsOnPurpose(&Finder->getASTContext(), &Node);
}

} // namespace

SuspiciousMissingCommaCheck::SuspiciousMissingCommaCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SizeThreshold(Options.get("SizeThreshold", 5U)),
      RatioThreshold(std::stod(Options.get("RatioThreshold", ".2"))),
      MaxConcatenatedTokens(Options.get("MaxConcatenatedTokens", 5U)) {}

void SuspiciousMissingCommaCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SizeThreshold", SizeThreshold);
  Options.store(Opts, "RatioThreshold", std::to_string(RatioThreshold));
  Options.store(Opts, "MaxConcatenatedTokens", MaxConcatenatedTokens);
}

void SuspiciousMissingCommaCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConcatenatedStringLiteral =
      stringLiteral(isConcatenatedLiteral(MaxConcatenatedTokens)).bind("str");

  const auto StringsInitializerList =
      initListExpr(hasType(constantArrayType()),
                   has(ignoringParenImpCasts(expr(ConcatenatedStringLiteral))));

  Finder->addMatcher(StringsInitializerList.bind("list"), this);
}

void SuspiciousMissingCommaCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitializerList = Result.Nodes.getNodeAs<InitListExpr>("list");
  const auto *ConcatenatedLiteral =
      Result.Nodes.getNodeAs<StringLiteral>("str");
  assert(InitializerList && ConcatenatedLiteral);

  // Skip small arrays as they often generate false-positive.
  unsigned int Size = InitializerList->getNumInits();
  if (Size < SizeThreshold)
    return;

  // Count the number of occurence of concatenated string literal.
  unsigned int Count = 0;
  for (unsigned int i = 0; i < Size; ++i) {
    const Expr *Child = InitializerList->getInit(i)->IgnoreImpCasts();
    if (const auto *Literal = dyn_cast<StringLiteral>(Child)) {
      if (Literal->getNumConcatenated() > 1)
        ++Count;
    }
  }

  // Warn only when concatenation is not common in this initializer list.
  // The current threshold is set to less than 1/5 of the string literals.
  if (double(Count) / Size > RatioThreshold)
    return;

  diag(ConcatenatedLiteral->getBeginLoc(),
       "suspicious string literal, probably missing a comma");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
