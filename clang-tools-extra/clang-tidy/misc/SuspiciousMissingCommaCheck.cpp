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
namespace misc {

namespace {

AST_MATCHER(StringLiteral, isConcatenatedLiteral) {
  return Node.getNumConcatenated() > 1;
}

}  // namespace

SuspiciousMissingCommaCheck::SuspiciousMissingCommaCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SizeThreshold(Options.get("SizeThreshold", 5U)),
      RatioThreshold(std::stod(Options.get("RatioThreshold", ".2"))) {}

void SuspiciousMissingCommaCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SizeThreshold", SizeThreshold);
  Options.store(Opts, "RatioThreshold", std::to_string(RatioThreshold));
}

void SuspiciousMissingCommaCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConcatenatedStringLiteral =
      stringLiteral(isConcatenatedLiteral()).bind("str");

  const auto StringsInitializerList =
      initListExpr(hasType(constantArrayType()),
                   has(expr(ignoringImpCasts(ConcatenatedStringLiteral))));

  Finder->addMatcher(StringsInitializerList.bind("list"), this);
}

void SuspiciousMissingCommaCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitializerList = Result.Nodes.getNodeAs<InitListExpr>("list");
  const auto *ConcatenatedLiteral = Result.Nodes.getNodeAs<Expr>("str");
  assert(InitializerList && ConcatenatedLiteral);

  // Skip small arrays as they often generate false-positive.
  unsigned int Size = InitializerList->getNumInits();
  if (Size < SizeThreshold) return;

  // Count the number of occurence of concatenated string literal.
  unsigned int Count = 0;
  for (unsigned int i = 0; i < Size; ++i) {
    const Expr *Child = InitializerList->getInit(i)->IgnoreImpCasts();
    if (const auto *Literal = dyn_cast<StringLiteral>(Child)) {
      if (Literal->getNumConcatenated() > 1) ++Count;
    }
  }

  // Warn only when concatenation is not common in this initializer list.
  // The current threshold is set to less than 1/5 of the string literals.
  if (double(Count) / Size > RatioThreshold) return;

  diag(ConcatenatedLiteral->getLocStart(),
       "suspicious string literal, probably missing a comma");
}

}  // namespace misc
}  // namespace tidy
}  // namespace clang
