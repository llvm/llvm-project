//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseVaOptCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
class VaOptPPCallbacks : public PPCallbacks {
public:
  VaOptPPCallbacks(Preprocessor *PP, UseVaOptCheck *Check)
      : PP(PP), Check(Check) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    const MacroInfo *MI = MD->getMacroInfo();
    if (!MI->isVariadic())
      return;

    std::optional<Token> PrevComma;
    bool PrevHashHash = false;
    for (const Token Tok : MI->tokens()) {
      if (PrevHashHash) {
        assert(PrevComma);
        if (const auto *II = Tok.getIdentifierInfo();
            II && II->getName() == "__VA_ARGS__") {
          // FIXME: The replacement really should be " __VA_OPT__(,)
          // __VA_ARGS__", but this breaks the fixit which removes the , in that
          // case o_O.
          Check->diag(Tok.getLocation(),
                      "Use __VA_OPT__ instead of GNU extension to __VA_ARGS__")
              << FixItHint::CreateReplacement(
                     SourceRange(PrevComma->getLocation(), Tok.getLocation()),
                     " __VA_OPT__(',') __VA_ARGS__");
        }
        PrevComma = std::nullopt;
        PrevHashHash = false;
      } else if (PrevComma) {
        assert(!PrevHashHash);
        if (Tok.is(tok::hashhash))
          PrevHashHash = true;
        else
          PrevComma = std::nullopt;
      } else if (Tok.is(tok::comma)) {
        assert(!PrevHashHash);
        PrevComma = Tok;
      }
    }
  }

private:
  Preprocessor *PP;
  UseVaOptCheck *Check;
};
} // namespace

void UseVaOptCheck::registerPPCallbacks(const SourceManager &SM,
                                        Preprocessor *PP,
                                        Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<VaOptPPCallbacks>(PP, this));
}

} // namespace clang::tidy::modernize
