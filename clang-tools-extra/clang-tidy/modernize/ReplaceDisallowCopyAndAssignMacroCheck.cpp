//===--- ReplaceDisallowCopyAndAssignMacroCheck.cpp - clang-tidy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceDisallowCopyAndAssignMacroCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

namespace clang::tidy::modernize {

namespace {

class ReplaceDisallowCopyAndAssignMacroCallbacks : public PPCallbacks {
public:
  explicit ReplaceDisallowCopyAndAssignMacroCallbacks(
      ReplaceDisallowCopyAndAssignMacroCheck &Check, Preprocessor &PP)
      : Check(Check), PP(PP) {}

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    IdentifierInfo *Info = MacroNameTok.getIdentifierInfo();
    if (!Info || !Args || Args->getNumMacroArguments() != 1)
      return;
    if (Info->getName() != Check.getMacroName())
      return;
    // The first argument to the DISALLOW_COPY_AND_ASSIGN macro is expected to
    // be the class name.
    const Token *ClassNameTok = Args->getUnexpArgument(0);
    if (Args->ArgNeedsPreexpansion(ClassNameTok, PP))
      // For now we only support simple argument that don't need to be
      // pre-expanded.
      return;
    clang::IdentifierInfo *ClassIdent = ClassNameTok->getIdentifierInfo();
    if (!ClassIdent)
      return;

    std::string Replacement = llvm::formatv(
        R"cpp({0}(const {0} &) = delete;
const {0} &operator=(const {0} &) = delete{1})cpp",
        ClassIdent->getName(), shouldAppendSemi(Range) ? ";" : "");

    Check.diag(MacroNameTok.getLocation(),
               "prefer deleting copy constructor and assignment operator over "
               "using macro '%0'")
        << Check.getMacroName()
        << FixItHint::CreateReplacement(
               PP.getSourceManager().getExpansionRange(Range), Replacement);
  }

private:
  /// \returns \c true if the next token after the given \p MacroLoc is \b not a
  /// semicolon.
  bool shouldAppendSemi(SourceRange MacroLoc) {
    std::optional<Token> Next = Lexer::findNextToken(
        MacroLoc.getEnd(), PP.getSourceManager(), PP.getLangOpts());
    return !(Next && Next->is(tok::semi));
  }

  ReplaceDisallowCopyAndAssignMacroCheck &Check;
  Preprocessor &PP;
};
} // namespace

ReplaceDisallowCopyAndAssignMacroCheck::ReplaceDisallowCopyAndAssignMacroCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MacroName(Options.get("MacroName", "DISALLOW_COPY_AND_ASSIGN")) {}

void ReplaceDisallowCopyAndAssignMacroCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      ::std::make_unique<ReplaceDisallowCopyAndAssignMacroCallbacks>(
          *this, *ModuleExpanderPP));
}

void ReplaceDisallowCopyAndAssignMacroCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MacroName", MacroName);
}

} // namespace clang::tidy::modernize
