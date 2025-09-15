//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantPreprocessorCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang::tidy::readability {

namespace {
/// Information about an opening preprocessor directive.
struct PreprocessorEntry {
  SourceLocation Loc;
  /// Condition used after the preprocessor directive.
  std::string Condition;
};

const char WarningDescription[] =
    "nested redundant %select{#if|#ifdef|#ifndef}0; consider removing it";
const char NoteDescription[] = "previous %select{#if|#ifdef|#ifndef}0 was here";

class RedundantPreprocessorCallbacks : public PPCallbacks {
  enum DirectiveKind { DK_If = 0, DK_Ifdef = 1, DK_Ifndef = 2 };

public:
  explicit RedundantPreprocessorCallbacks(ClangTidyCheck &Check,
                                          Preprocessor &PP)
      : Check(Check), PP(PP) {}

  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override {
    StringRef Condition =
        Lexer::getSourceText(CharSourceRange::getTokenRange(ConditionRange),
                             PP.getSourceManager(), PP.getLangOpts());
    checkMacroRedundancy(Loc, Condition, IfStack, DK_If, DK_If, true);
  }

  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MacroDefinition) override {
    std::string MacroName = PP.getSpelling(MacroNameTok);
    checkMacroRedundancy(Loc, MacroName, IfdefStack, DK_Ifdef, DK_Ifdef, true);
    checkMacroRedundancy(Loc, MacroName, IfndefStack, DK_Ifdef, DK_Ifndef,
                         false);
  }

  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MacroDefinition) override {
    std::string MacroName = PP.getSpelling(MacroNameTok);
    checkMacroRedundancy(Loc, MacroName, IfndefStack, DK_Ifndef, DK_Ifndef,
                         true);
    checkMacroRedundancy(Loc, MacroName, IfdefStack, DK_Ifndef, DK_Ifdef,
                         false);
  }

  void Endif(SourceLocation Loc, SourceLocation IfLoc) override {
    if (!IfStack.empty() && IfLoc == IfStack.back().Loc)
      IfStack.pop_back();
    if (!IfdefStack.empty() && IfLoc == IfdefStack.back().Loc)
      IfdefStack.pop_back();
    if (!IfndefStack.empty() && IfLoc == IfndefStack.back().Loc)
      IfndefStack.pop_back();
  }

private:
  void checkMacroRedundancy(SourceLocation Loc, StringRef MacroName,
                            SmallVector<PreprocessorEntry, 4> &Stack,
                            DirectiveKind WarningKind, DirectiveKind NoteKind,
                            bool Store) {
    if (PP.getSourceManager().isInMainFile(Loc)) {
      for (const auto &Entry : Stack) {
        if (Entry.Condition == MacroName) {
          Check.diag(Loc, WarningDescription) << WarningKind;
          Check.diag(Entry.Loc, NoteDescription, DiagnosticIDs::Note)
              << NoteKind;
        }
      }
    }

    if (Store)
      // This is an actual directive to be remembered.
      Stack.push_back({Loc, std::string(MacroName)});
  }

  ClangTidyCheck &Check;
  Preprocessor &PP;
  SmallVector<PreprocessorEntry, 4> IfStack;
  SmallVector<PreprocessorEntry, 4> IfdefStack;
  SmallVector<PreprocessorEntry, 4> IfndefStack;
};
} // namespace

void RedundantPreprocessorCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      ::std::make_unique<RedundantPreprocessorCallbacks>(*this, *PP));
}

} // namespace clang::tidy::readability
