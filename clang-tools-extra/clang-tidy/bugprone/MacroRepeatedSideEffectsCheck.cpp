//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroRepeatedSideEffectsCheck.h"
#include "clang/Basic/Builtins.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>
#include <stack>

namespace clang::tidy::bugprone {

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

  bool isNonZeroLiteral(const Token &T) const;

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
  if (llvm::any_of(MI->tokens(), [](const Token &T) {
        return T.isOneOf(tok::kw_if, tok::kw_else, tok::kw_switch, tok::kw_case,
                         tok::kw_break, tok::kw_while, tok::kw_do, tok::kw_for,
                         tok::kw_continue, tok::kw_goto, tok::kw_return);
      }))
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
  int ParenDepth = 0;
  // Paren depth of the false branch of "(1 ? a : b)" while we are skipping it.
  std::optional<int> DeadBranchDepth;
  // Count when "?" is reached. The "Current" will get this value when the ":"
  // is reached. Also remembers if the condition was a non-zero literal.
  std::stack<std::pair<unsigned, bool>,
             SmallVector<std::pair<unsigned, bool>, 8>>
      CountAtQuestion;
  const ArrayRef<Token> Tokens = MI->tokens();
  for (const auto [I, T] : llvm::enumerate(Tokens)) {
    if (T.is(tok::l_paren))
      ParenDepth++;
    else if (T.is(tok::r_paren))
      ParenDepth--;

    // The false branch of "(1 ? a : b)" is never evaluated. Skip it until the
    // closing parenthesis or a comma.
    if (DeadBranchDepth) {
      if (ParenDepth > *DeadBranchDepth ||
          (ParenDepth == *DeadBranchDepth && T.isNot(tok::comma)))
        continue;
      DeadBranchDepth.reset();
    }

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
      CountAtQuestion.emplace(Current, I >= 2 &&
                                           Tokens[I - 2].is(tok::l_paren) &&
                                           isNonZeroLiteral(Tokens[I - 1]));
    } else if (T.is(tok::colon)) {
      if (CountAtQuestion.empty())
        return 0;
      bool SkipFalseBranch = false;
      std::tie(Current, SkipFalseBranch) = CountAtQuestion.top();
      CountAtQuestion.pop();
      if (SkipFalseBranch) {
        DeadBranchDepth = ParenDepth;
        continue;
      }
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

    const IdentifierInfo *TII = T.getIdentifierInfo();
    // If not existent, skip it.
    if (TII == nullptr)
      continue;

    // Arguments of sizeof and alignof are not evaluated.
    if (T.isOneOf(tok::kw_sizeof, tok::kw_alignof, tok::kw__Alignof)) {
      SkipParen = true;
      continue;
    }

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
      Max = std::max(Max, Current);
    }
  }
  return Max;
}

bool MacroRepeatedPPCallbacks::isNonZeroLiteral(const Token &T) const {
  if (T.isNot(tok::numeric_constant))
    return false;
  SmallString<16> Buffer;
  uint64_t Value = 0;
  return !PP.getSpelling(T, Buffer).getAsInteger(0, Value) && Value != 0;
}

bool MacroRepeatedPPCallbacks::hasSideEffects(
    const Token *ResultArgToks) const {
  for (; ResultArgToks->isNot(tok::eof); ++ResultArgToks)
    if (ResultArgToks->isOneOf(tok::plusplus, tok::minusminus))
      return true;
  return false;
}

void MacroRepeatedSideEffectsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(::std::make_unique<MacroRepeatedPPCallbacks>(*this, *PP));
}

} // namespace clang::tidy::bugprone
