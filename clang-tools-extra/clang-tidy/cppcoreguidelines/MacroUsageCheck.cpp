//===--- MacroUsageCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MacroUsageCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Regex.h"
#include <algorithm>

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

namespace {

bool isCapsOnly(StringRef Name) {
  return std::all_of(Name.begin(), Name.end(), [](const char c) {
    if (std::isupper(c) || std::isdigit(c) || c == '_')
      return true;
    return false;
  });
}

class MacroUsageCallbacks : public PPCallbacks {
public:
  MacroUsageCallbacks(MacroUsageCheck *Check, StringRef RegExp, bool CapsOnly)
      : Check(Check), RegExp(RegExp), CheckCapsOnly(CapsOnly) {}
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    if (MD->getMacroInfo()->isUsedForHeaderGuard() ||
        MD->getMacroInfo()->getNumTokens() == 0)
      return;

    StringRef MacroName = MacroNameTok.getIdentifierInfo()->getName();
    if (!CheckCapsOnly && !llvm::Regex(RegExp).match(MacroName))
      Check->warnMacro(MD);

    if (CheckCapsOnly && !isCapsOnly(MacroName))
      Check->warnNaming(MD);
  }

private:
  MacroUsageCheck *Check;
  StringRef RegExp;
  bool CheckCapsOnly;
};
} // namespace

void MacroUsageCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedRegexp", AllowedRegexp);
  Options.store(Opts, "CheckCapsOnly", CheckCapsOnly);
}

void MacroUsageCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  if (!getLangOpts().CPlusPlus11)
    return;

  Compiler.getPreprocessor().addPPCallbacks(
      llvm::make_unique<MacroUsageCallbacks>(this, AllowedRegexp,
                                             CheckCapsOnly));
}

void MacroUsageCheck::warnMacro(const MacroDirective *MD) {
  StringRef Message =
      "macro used to declare a constant; consider using a 'constexpr' "
      "constant";

  /// A variadic macro is function-like at the same time. Therefore variadic
  /// macros are checked first and will be excluded for the function-like
  /// diagnostic.
  if (MD->getMacroInfo()->isVariadic())
    Message = "variadic macro used; consider using a 'constexpr' "
              "variadic template function";
  else if (MD->getMacroInfo()->isFunctionLike())
    Message = "function-like macro used; consider a 'constexpr' template "
              "function";

  diag(MD->getLocation(), Message);
}

void MacroUsageCheck::warnNaming(const MacroDirective *MD) {
  diag(MD->getLocation(), "macro definition does not define the macro name "
                          "using all uppercase characters");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
