//===--- BadSignalToKillThreadCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BadSignalToKillThreadCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void BadSignalToKillThreadCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(allOf(callee(functionDecl(hasName("::pthread_kill"))),
                     argumentCountIs(2)),
               hasArgument(1, integerLiteral().bind("integer-literal")))
          .bind("thread-kill"),
      this);
}

static Preprocessor *PP;

void BadSignalToKillThreadCheck::check(const MatchFinder::MatchResult &Result) {
  const auto IsSigterm = [](const auto &KeyValue) -> bool {
    return KeyValue.first->getName() == "SIGTERM";
  };
  const auto TryExpandAsInteger =
      [](Preprocessor::macro_iterator It) -> Optional<unsigned> {
    if (It == PP->macro_end())
      return llvm::None;
    const MacroInfo *MI = PP->getMacroInfo(It->first);
    const Token &T = MI->tokens().back();
    StringRef ValueStr = StringRef(T.getLiteralData(), T.getLength());

    llvm::APInt IntValue;
    constexpr unsigned AutoSenseRadix = 0;
    if (ValueStr.getAsInteger(AutoSenseRadix, IntValue))
      return llvm::None;
    return IntValue.getZExtValue();
  };

  const auto SigtermMacro = llvm::find_if(PP->macros(), IsSigterm);

  if (!SigtermValue && !(SigtermValue = TryExpandAsInteger(SigtermMacro)))
    return;

  const auto *MatchedExpr = Result.Nodes.getNodeAs<Expr>("thread-kill");
  const auto *MatchedIntLiteral =
      Result.Nodes.getNodeAs<IntegerLiteral>("integer-literal");
  if (MatchedIntLiteral->getValue() == *SigtermValue) {
    diag(MatchedExpr->getBeginLoc(),
         "thread should not be terminated by raising the 'SIGTERM' signal");
  }
}

void BadSignalToKillThreadCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *pp, Preprocessor *ModuleExpanderPP) {
  PP = pp;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
