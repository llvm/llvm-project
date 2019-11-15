//===--- UseUsingCheck.cpp - clang-tidy------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseUsingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

UseUsingCheck::UseUsingCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void UseUsingCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;
  Finder->addMatcher(typedefDecl(unless(isInstantiated())).bind("typedef"),
                     this);
}

// Checks if 'typedef' keyword can be removed - we do it only if
// it is the only declaration in a declaration chain.
static bool CheckRemoval(SourceManager &SM, SourceLocation StartLoc,
                         ASTContext &Context) {
  assert(StartLoc.isFileID() && "StartLoc must not be in a macro");
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(StartLoc);
  StringRef File = SM.getBufferData(LocInfo.first);
  const char *TokenBegin = File.data() + LocInfo.second;
  Lexer DeclLexer(SM.getLocForStartOfFile(LocInfo.first), Context.getLangOpts(),
                  File.begin(), TokenBegin, File.end());

  Token Tok;
  int NestingLevel = 0; // Parens, braces, and square brackets
  int AngleBracketLevel = 0;
  bool FoundTypedef = false;

  while (!DeclLexer.LexFromRawLexer(Tok) && !Tok.is(tok::semi)) {
    switch (Tok.getKind()) {
    case tok::l_brace:
      if (NestingLevel == 0 && AngleBracketLevel == 0) {
        // At top level, this might be the `typedef struct {...} T;` case.
        // Inside parens, square brackets, or angle brackets it's not.
        return false;
      }
      ++NestingLevel;
      break;
    case tok::l_paren:
    case tok::l_square:
      ++NestingLevel;
      break;
    case tok::r_brace:
    case tok::r_paren:
    case tok::r_square:
      --NestingLevel;
      break;
    case tok::less:
      // If not nested in paren/brace/square bracket, treat as opening angle bracket.
      if (NestingLevel == 0)
        ++AngleBracketLevel;
      break;
    case tok::greater:
      // Per C++ 17 Draft N4659, Section 17.2/3
      //   https://timsong-cpp.github.io/cppwp/n4659/temp.names#3:
      // "When parsing a template-argument-list, the first non-nested > is
      // taken as the ending delimiter rather than a greater-than operator."
      // If not nested in paren/brace/square bracket, treat as closing angle bracket.
      if (NestingLevel == 0)
        --AngleBracketLevel;
      break;
    case tok::comma:
      if (NestingLevel == 0 && AngleBracketLevel == 0) {
        // If there is a non-nested comma we have two or more declarations in this chain.
        return false;
      }
      break;
    case tok::raw_identifier:
      if (Tok.getRawIdentifier() == "typedef") {
        FoundTypedef = true;
      }
      break;
    default:
      break;
    }
  }

  // Sanity check against weird macro cases.
  return FoundTypedef;
}

void UseUsingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<TypedefDecl>("typedef");
  if (MatchedDecl->getLocation().isInvalid())
    return;

  auto &Context = *Result.Context;
  auto &SM = *Result.SourceManager;

  SourceLocation StartLoc = MatchedDecl->getBeginLoc();

  if (StartLoc.isMacroID() && IgnoreMacros)
    return;

  auto Diag = diag(StartLoc, "use 'using' instead of 'typedef'");

  // do not fix if there is macro or array
  if (MatchedDecl->getUnderlyingType()->isArrayType() || StartLoc.isMacroID())
    return;

  if (CheckRemoval(SM, StartLoc, Context)) {
    auto printPolicy = PrintingPolicy(getLangOpts());
    printPolicy.SuppressScope = true;
    printPolicy.ConstantArraySizeAsWritten = true;
    printPolicy.UseVoidForZeroParams = false;

    Diag << FixItHint::CreateReplacement(
        MatchedDecl->getSourceRange(),
        "using " + MatchedDecl->getNameAsString() + " = " +
            MatchedDecl->getUnderlyingType().getAsString(printPolicy));
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
