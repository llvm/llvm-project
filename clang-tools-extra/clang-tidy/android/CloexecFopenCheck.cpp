//===--- CloexecFopenCheck.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.  //
//===----------------------------------------------------------------------===//

#include "CloexecFopenCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

namespace {
static const char MODE = 'e';

// Build the replace text. If it's string constant, add 'e' directly in the end
// of the string. Else, add "e".
std::string BuildReplaceText(const Expr *Arg, const SourceManager &SM,
                             const LangOptions &LangOpts) {
  if (Arg->getLocStart().isMacroID())
    return (Lexer::getSourceText(
                CharSourceRange::getTokenRange(Arg->getSourceRange()), SM,
                LangOpts) +
            " \"" + Twine(MODE) + "\"")
        .str();

  StringRef SR = cast<StringLiteral>(Arg->IgnoreParenCasts())->getString();
  return ("\"" + SR + Twine(MODE) + "\"").str();
}
} // namespace

void CloexecFopenCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));

  Finder->addMatcher(
      callExpr(callee(functionDecl(isExternC(), returns(asString("FILE *")),
                                   hasName("fopen"),
                                   hasParameter(0, CharPointerType),
                                   hasParameter(1, CharPointerType))
                          .bind("funcDecl")))
          .bind("fopenFn"),
      this);
}

void CloexecFopenCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("fopenFn");
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("funcDecl");
  const Expr *ModeArg = MatchedCall->getArg(1);

  // Check if the 'e' may be in the mode string.
  const auto *ModeStr = dyn_cast<StringLiteral>(ModeArg->IgnoreParenCasts());
  if (!ModeStr || (ModeStr->getString().find(MODE) != StringRef::npos))
    return;

  const std::string &ReplacementText = BuildReplaceText(
      ModeArg, *Result.SourceManager, Result.Context->getLangOpts());

  diag(ModeArg->getLocStart(), "use %0 mode 'e' to set O_CLOEXEC")
      << FD
      << FixItHint::CreateReplacement(ModeArg->getSourceRange(),
                                      ReplacementText);
}

} // namespace android
} // namespace tidy
} // namespace clang
