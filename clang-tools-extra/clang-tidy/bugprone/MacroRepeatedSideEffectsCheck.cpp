//===--- MacroRepeatedSideEffectsCheck.cpp - clang-tidy--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroRepeatedSideEffectsCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
class MacroRepeatedPPCallbacks : public PPCallbacks {
public:
  MacroRepeatedPPCallbacks(ClangTidyCheck &Check, Preprocessor &PP)
      : Check(Check), PP(PP) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;

private:
  ClangTidyCheck &Check;
  Preprocessor &PP;

  unsigned countArgumentExpansions(const MacroInfo *MI,
                                   const IdentifierInfo *Arg) const;

  bool hasSideEffects(const Token *ResultArgToks) const;
};
} // End of anonymous namespace.

void MacroRepeatedPPCallbacks::MacroExpands(const Token &MacroNameTok,
                                            const MacroDefinition &MD,
                                            SourceRange Range,
                                            const MacroArgs *Args) {
  // Ignore macro argument expansions.
  if (!Range.getBegin().isFileID())
    return;

  const MacroInfo *MI = MD.getMacroInfo();

  // Bail out if the contents of the macro are containing keywords that are
  // making the macro too complex.
  if (std::find_if(
          MI->tokens().begin(), MI->tokens().end(), [](const Token &T) {
            return T.isOneOf(tok::kw_if, tok::kw_else, tok::kw_switch,
                             tok::kw_case, tok::kw_break, tok::kw_while,
                             tok::kw_do, tok::kw_for, tok::kw_continue,
                             tok::kw_goto, tok::kw_return);
          }) != MI->tokens().end())
    return;

  for (unsigned ArgNo = 0U; ArgNo < MI->getNumParams(); ++ArgNo) {
    const IdentifierInfo *Arg = *(MI->param_begin() + ArgNo);
    const Token *ResultArgToks = Args->getUnexpArgument(ArgNo);

    if (hasSideEffects(ResultArgToks) &&
        countArgumentExpansions(MI, Arg) >= 2) {
      Check.diag(ResultArgToks->getLocation(),
                 "side effects in the %ordinal0 macro argument %1 are "
                 "repeated in macro expansion")
          << (ArgNo + 1) << Arg;
      Check.diag(MI->getDefinitionLoc(), "macro %0 defined here",
                 DiagnosticIDs::Note)
          << MacroNameTok.getIdentifierInfo();
    }
  }
}

unsigned MacroRepeatedPPCallbacks::countArgumentExpansions(
    const MacroInfo *MI, const IdentifierInfo *Arg) const {
  // Current argument count. When moving forward to a different control-flow
  // path this can decrease.
  unsigned Current = 0;
  // Max argument count.
  unsigned Max = 0;
  bool SkipParen = false;
  int SkipParenCount = 0;
  // Has a __builtin_constant_p been found?
  bool FoundBuiltin = false;
  bool PrevTokenIsHash = false;
  // Count when "?" is reached. The "Current" will get this value when the ":"
  // is reached.
  std::stack<unsigned, SmallVector<unsigned, 8>> CountAtQuestion;
  for (const auto &T : MI->tokens()) {
    // The result of __builtin_constant_p(x) is 0 if x is a macro argument
    // with side effects. If we see a __builtin_constant_p(x) followed by a
    // "?" "&&" or "||", then we need to reason about control flow to report
    // warnings correctly. Until such reasoning is added, bail out when this
    // happens.
    if (FoundBuiltin && T.isOneOf(tok::question, tok::ampamp, tok::pipepipe))
      return Max;

    // Skip stringified tokens.
    if (T.is(tok::hash)) {
      PrevTokenIsHash = true;
      continue;
    }
    if (PrevTokenIsHash) {
      PrevTokenIsHash = false;
      continue;
    }

    // Handling of ? and :.
    if (T.is(tok::question)) {
      CountAtQuestion.push(Current);
    } else if (T.is(tok::colon)) {
      if (CountAtQuestion.empty())
        return 0;
      Current = CountAtQuestion.top();
      CountAtQuestion.pop();
    }

    // If current token is a parenthesis, skip it.
    if (SkipParen) {
      if (T.is(tok::l_paren))
        SkipParenCount++;
      else if (T.is(tok::r_paren))
        SkipParenCount--;
      SkipParen = (SkipParenCount != 0);
      if (SkipParen)
        continue;
    }

    IdentifierInfo *TII = T.getIdentifierInfo();
    // If not existent, skip it.
    if (TII == nullptr)
      continue;

    // If a __builtin_constant_p is found within the macro definition, don't
    // count arguments inside the parentheses and remember that it has been
    // seen in case there are "?", "&&" or "||" operators later.
    if (TII->getBuiltinID() == Builtin::BI__builtin_constant_p) {
      FoundBuiltin = true;
      SkipParen = true;
      continue;
    }

    // If another macro is found within the macro definition, skip the macro
    // and the eventual arguments.
    if (TII->hasMacroDefinition()) {
      const MacroInfo *M = PP.getMacroDefinition(TII).getMacroInfo();
      if (M != nullptr && M->isFunctionLike())
        SkipParen = true;
      continue;
    }

    // Count argument.
    if (TII == Arg) {
      Current++;
      if (Current > Max)
        Max = Current;
    }
  }
  return Max;
}

bool MacroRepeatedPPCallbacks::hasSideEffects(
    const Token *ResultArgToks) const {
  for (; ResultArgToks->isNot(tok::eof); ++ResultArgToks) {
    if (ResultArgToks->isOneOf(tok::plusplus, tok::minusminus))
      return true;
  }
  return false;
}

void MacroRepeatedSideEffectsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(::llvm::make_unique<MacroRepeatedPPCallbacks>(*this, *PP));
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
