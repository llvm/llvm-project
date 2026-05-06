//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatvStringCheck.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {

struct ParseResult {
  llvm::SmallVector<unsigned, 4> Indices;
  unsigned MaxIndex = 0;
};

} // namespace

static llvm::Expected<ParseResult> parseFormatvString(llvm::StringRef Fmt) {
  ParseResult Result;
  unsigned NextAutoIndex = 0;
  bool HasAutomatic = false;
  bool HasExplicit = false;

  while (!Fmt.empty()) {
    const size_t OpenBrace = Fmt.find('{');
    if (OpenBrace == llvm::StringRef::npos)
      break;

    Fmt = Fmt.drop_front(OpenBrace);

    // Handle escaped braces '{{'.
    if (Fmt.size() > 1 && Fmt[1] == '{') {
      Fmt = Fmt.drop_front(2);
      continue;
    }

    // Find the closing '}'.
    const size_t CloseBrace = Fmt.find('}');
    if (CloseBrace == llvm::StringRef::npos)
      return llvm::createStringError("unterminated brace in format string");

    // Extract the content between braces.
    const llvm::StringRef Content = Fmt.substr(1, CloseBrace - 1);
    Fmt = Fmt.drop_front(CloseBrace + 1);

    // Parse the replacement field: [index] ["," layout] [":" format]
    llvm::StringRef IndexStr = Content;

    // Strip layout and format parts for index parsing.
    const size_t CommaPos = Content.find(',');
    const size_t ColonPos = Content.find(':');
    if (CommaPos != llvm::StringRef::npos)
      IndexStr = Content.substr(0, CommaPos);
    else if (ColonPos != llvm::StringRef::npos)
      IndexStr = Content.substr(0, ColonPos);

    IndexStr = IndexStr.trim();

    unsigned Index = 0;
    if (IndexStr.empty()) {
      Index = NextAutoIndex++;
      HasAutomatic = true;
    } else {
      if (IndexStr.getAsInteger(10, Index))
        return llvm::createStringError("invalid replacement index");
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
  // Always check llvm::formatv (both overloads).
  Functions["llvm::formatv"] = 0;

  // Parse "name1:idx1;name2:idx2;..." from AdditionalFunctions.
  llvm::StringRef Input(AdditionalFunctions);
  while (!Input.empty()) {
    auto [Entry, Rest] = Input.split(';');
    Input = Rest;
    if (Entry.empty())
      continue;
    auto [Name, IdxStr] = Entry.rsplit(':');
    unsigned Idx = 0;
    if (Name.empty() || IdxStr.empty() || IdxStr.getAsInteger(10, Idx)) {
      configurationDiag("invalid entry '%0' in option AdditionalFunctions, "
                        "expected 'fully::qualified::name:fmt_arg_index'")
          << Entry;
      continue;
    }
    Functions[Name] = Idx;
  }
}

void FormatvStringCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AdditionalFunctions", AdditionalFunctions);
}

void FormatvStringCheck::registerMatchers(MatchFinder *Finder) {
  // Build a matcher for all configured function names.
  std::vector<llvm::StringRef> Names;
  Names.reserve(Functions.size());
  llvm::copy(Functions.keys(), std::back_inserter(Names));

  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName(Names)))).bind("call"), this);
}

void FormatvStringCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  if (!Call || Call->getNumArgs() == 0)
    return;

  const auto *FD = Call->getDirectCallee();
  if (!FD)
    return;

  // Look up the format string parameter index for this function.
  const std::string QualName = FD->getQualifiedNameAsString();
  assert(Functions.contains(QualName) &&
         "matched function not in Functions map");
  unsigned FmtArgIdx = Functions.lookup(QualName);

  // For llvm::formatv, also handle the (bool, const char*, ...) overload.
  if (QualName == "llvm::formatv" && FD->getNumParams() > 0 &&
      FD->getParamDecl(0)->getType()->isBooleanType())
    FmtArgIdx = 1;

  if (Call->getNumArgs() <= FmtArgIdx)
    return;

  // Extract the format string literal.
  const Expr *FmtArg = Call->getArg(FmtArgIdx)->IgnoreParenImpCasts();
  const auto *FmtLiteral = dyn_cast<StringLiteral>(FmtArg);
  if (!FmtLiteral)
    return;

  const llvm::StringRef FmtStr = FmtLiteral->getString();
  const unsigned FirstArgIdx = FmtArgIdx + 1;
  const int NumArgs = Call->getNumArgs() - FirstArgIdx;

  auto ParsedOrErr = parseFormatvString(FmtStr);
  if (!ParsedOrErr) {
    diag(FmtLiteral->getBeginLoc(), "formatv() %0")
        << llvm::toString(ParsedOrErr.takeError());
    return;
  }

  const ParseResult &Parsed = *ParsedOrErr;
  const int NumRequiredArgs = Parsed.Indices.empty() ? 0 : Parsed.MaxIndex + 1;

  if (NumRequiredArgs != NumArgs) {
    diag(FmtLiteral->getBeginLoc(),
         "formatv() format string requires %0 argument(s), but %1 "
         "argument(s) were provided")
        << NumRequiredArgs << NumArgs;
    return;
  }

  // Check for holes in indices.
  if (!Parsed.Indices.empty()) {
    llvm::SmallBitVector UsedIndices(NumRequiredArgs);
    for (const unsigned Index : Parsed.Indices)
      UsedIndices.set(Index);

    const int UnusedIndex = UsedIndices.find_first_unset();
    if (0 <= UnusedIndex && UnusedIndex < NumRequiredArgs) {
      // Point to the unused argument.
      const Expr *UnusedArg = Call->getArg(FirstArgIdx + UnusedIndex);
      diag(UnusedArg->getBeginLoc(),
           "formatv() format string does not use argument at index %0")
          << UnusedIndex;
      return;
    }
  }
}

} // namespace clang::tidy::llvm_check
