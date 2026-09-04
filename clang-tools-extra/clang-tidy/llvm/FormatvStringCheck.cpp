//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatvStringCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {

struct ParseResult {
  SmallVector<unsigned, 4> Indices;
  unsigned MaxIndex = 0;
};

} // namespace

static Expected<ParseResult> parseFormatvString(StringRef Fmt) {
  ParseResult Result;
  unsigned NextAutoIndex = 0;
  bool HasAutomatic = false;
  bool HasExplicit = false;

  while (!Fmt.empty()) {
    const size_t OpenBrace = Fmt.find('{');
    if (OpenBrace == StringRef::npos)
      break;

    Fmt = Fmt.drop_front(OpenBrace);

    // Handle escaped braces '{{'.
    if (Fmt.consume_front("{{"))
      continue;

    // Find the closing '}'.
    const size_t CloseBrace = Fmt.find('}');
    if (CloseBrace == StringRef::npos)
      return llvm::createStringError("unterminated brace in format string");

    // Extract the content between braces.
    const StringRef Content = Fmt.substr(1, CloseBrace - 1);
    Fmt = Fmt.drop_front(CloseBrace + 1);

    // Parse the replacement field: [index] ["," layout] [":" format]
    StringRef IndexStr = Content.substr(0, Content.find_first_of(",:"));

    IndexStr = IndexStr.trim();

    unsigned Index = 0;
    if (IndexStr.empty()) {
      Index = NextAutoIndex++;
      HasAutomatic = true;
    } else {
      if (IndexStr.getAsInteger(10, Index))
        return llvm::createStringError(
            "invalid replacement index in format string");
      HasExplicit = true;
    }

    Result.Indices.push_back(Index);
    Result.MaxIndex = std::max(Result.MaxIndex, Index);
  }

  if (HasAutomatic && HasExplicit)
    return llvm::createStringError(
        "format string mixes automatic and explicit indices");

  return Result;
}

FormatvStringCheck::FormatvStringCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AdditionalFunctions(Options.get("AdditionalFunctions", "")) {
  Functions = utils::options::parseStringList(AdditionalFunctions);
  Functions.emplace_back("::llvm::formatv");
  Functions.emplace_back("::llvm::createStringErrorV");
}

void FormatvStringCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AdditionalFunctions", AdditionalFunctions);
}

void FormatvStringCheck::registerMatchers(MatchFinder *Finder) {
  // Build a matcher for all configured function names.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName(Functions),
                                   ast_matchers::isTemplateInstantiation())),
               argumentCountAtLeast(1))
          .bind("call"),
      this);
}

void FormatvStringCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  assert(Call && Call->getNumArgs() > 0);

  const auto *FD = Call->getDirectCallee();
  assert(FD);

  // Find the format string index from the template signature: it's the
  // parameter immediately before the trailing parameter pack.
  const FunctionDecl *TemplateDecl = FD;
  if (const FunctionTemplateDecl *Primary = FD->getPrimaryTemplate())
    TemplateDecl = Primary->getTemplatedDecl();

  const unsigned NumDeclParams = TemplateDecl->getNumParams();
  if (NumDeclParams < 2)
    return;

  const unsigned PackParamIndex = NumDeclParams - 1;
  if (!TemplateDecl->getParamDecl(PackParamIndex)->isParameterPack())
    return;

  const unsigned FmtStringIndex = PackParamIndex - 1;

  if (Call->getNumArgs() <= FmtStringIndex)
    return;

  // Extract the format string literal.
  const Expr *FmtArg = Call->getArg(FmtStringIndex)->IgnoreParenImpCasts();
  const auto *FmtLiteral = dyn_cast<StringLiteral>(FmtArg);
  if (!FmtLiteral)
    return;

  const StringRef FmtString = FmtLiteral->getString();
  const int NumFmtArgs = Call->getNumArgs() - PackParamIndex;

  auto ParsedOrErr = parseFormatvString(FmtString);
  if (!ParsedOrErr) {
    diag(FmtLiteral->getBeginLoc(), toString(ParsedOrErr.takeError()));
    return;
  }

  const ParseResult &Parsed = *ParsedOrErr;
  const int NumRequiredArgs = Parsed.Indices.empty() ? 0 : Parsed.MaxIndex + 1;

  if (NumRequiredArgs > NumFmtArgs) {
    diag(FmtLiteral->getBeginLoc(),
         "format string requires %0 argument%s0, but %1 argument%s1 "
         "%plural{1:was|:were}1 provided")
        << NumRequiredArgs << NumFmtArgs;
    return;
  }

  // Check for unused arguments: both indices not referenced by the format
  // string, and trailing arguments beyond what the format string requires.
  llvm::SmallBitVector UnusedIndices(NumFmtArgs, true);
  for (const unsigned Index : Parsed.Indices)
    UnusedIndices.reset(Index);

  for (const auto UnusedIndex : UnusedIndices.set_bits()) {
    const Expr *UnusedArg = Call->getArg(PackParamIndex + UnusedIndex);
    diag(UnusedArg->getBeginLoc(), "argument unused in format string");
  }
}

} // namespace clang::tidy::llvm_check
