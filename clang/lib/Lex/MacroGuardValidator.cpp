
#include "clang/Lex/MacroGuardValidator.h"

SmallVector<const IdentifierInfo *, 2> ArgsToEnclosedForMacroGuardValidator;

void MacroGuardValidator::MacroDefined(const Token &MacroNameTok,
                                       const MacroDirective *MD) {
  // Get macro info
  auto MI = MD->getMacroInfo();
  if (MI->tokens_empty()) {
    return;
  }

  for (auto II : ArgsToEnclosedForMacroGuardValidator) {
    // First, check if this macro really has this argument
    bool FindParam = false;
    for (auto MII : MI->params()) {
      if (MII == II) {
        FindParam = true;
      }
    }

    if (!FindParam) {
      // Not found macro param
      auto MacroNameII = MacroNameTok.getIdentifierInfo();
      assert(MacroNameII);
      auto NameTokLoc = MacroNameTok.getLocation();
      llvm::errs() << "[WARNING] Can't find argument '" << II->getName()
                   << "' ";
      llvm::errs() << "at macro '" << MacroNameII->getName() << "'(";
      llvm::errs() << NameTokLoc.printToString(SM) << ")\n";
      continue;
    }

    auto NumTokens = MI->getNumTokens();
    for (auto TokIdx = 0U; TokIdx < NumTokens; ++TokIdx) {
      auto CurTok = *(MI->tokens_begin() + TokIdx);
      if (CurTok.getIdentifierInfo() == II) {
        // Check if previous and successor Tokens are parenthesis
        if (TokIdx > 0 && TokIdx < NumTokens - 1) {
          auto PrevTok = *(MI->tokens_begin() + TokIdx - 1),
               NextTok = *(MI->tokens_begin() + TokIdx + 1);
          if (PrevTok.is(tok::l_paren) && NextTok.is(tok::r_paren)) {
            continue;
          }
        }

        // The argument is not enclosed
        auto CurTokLoc = CurTok.getLocation();
        llvm::errs() << "[WARNING] In " << CurTokLoc.printToString(SM) << ": ";
        llvm::errs() << "macro argument '" << II->getName()
                     << "' is not enclosed by parenthesis\n";
      }
    }
  }

  // Also clear the storage after one check
  ArgsToEnclosedForMacroGuardValidator.clear();
}
