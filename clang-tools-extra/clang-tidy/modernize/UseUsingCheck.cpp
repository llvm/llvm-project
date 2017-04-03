//===--- UseUsingCheck.cpp - clang-tidy------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
      IgnoreMacros(Options.get("IgnoreMacros", true)) {}

void UseUsingCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;
  Finder->addMatcher(typedefDecl().bind("typedef"), this);
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
  int ParenLevel = 0;
  bool FoundTypedef = false;

  while (!DeclLexer.LexFromRawLexer(Tok) && !Tok.is(tok::semi)) {
    switch (Tok.getKind()) {
    case tok::l_brace:
    case tok::r_brace:
      // This might be the `typedef struct {...} T;` case.
      return false;
    case tok::l_paren:
      ParenLevel++;
      break;
    case tok::r_paren:
      ParenLevel--;
      break;
    case tok::comma:
      if (ParenLevel == 0) {
        // If there is comma and we are not between open parenthesis then it is
        // two or more declarations in this chain.
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

  SourceLocation StartLoc = MatchedDecl->getLocStart();

  if (StartLoc.isMacroID() && IgnoreMacros)
    return;

  auto Diag =
      diag(StartLoc, "use 'using' instead of 'typedef'");

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
