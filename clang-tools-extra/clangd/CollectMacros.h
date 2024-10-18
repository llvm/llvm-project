//===--- CollectMacros.h -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COLLECTMACROS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COLLECTMACROS_H

#include "Protocol.h"
#include "SourceCode.h"
#include "index/SymbolID.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include <cstddef>
#include <string>

namespace clang {
namespace clangd {

struct MacroOccurrence {
  // Half-open range (end offset is exclusive) inside the main file.
  size_t StartOffset;
  size_t EndOffset;

  bool IsDefinition;
  // True if the occurence is used in a conditional directive, e.g. #ifdef MACRO
  bool InConditionalDirective;

  Range toRange(const SourceManager &SM) const;
};

struct MainFileMacros {
  llvm::StringSet<> Names;
  llvm::DenseMap<SymbolID, std::vector<MacroOccurrence>> MacroRefs;
  // Somtimes it is not possible to compute the SymbolID for the Macro, e.g. a
  // reference to an undefined macro. Store them separately, e.g. for semantic
  // highlighting.
  std::vector<MacroOccurrence> UnknownMacros;
  // Ranges skipped by the preprocessor due to being inactive.
  std::vector<Range> SkippedRanges;
};

/// Collects macro references (e.g. definitions, expansions) in the main file.
/// It is used to:
///  - collect macros in the preamble section of the main file (in Preamble.cpp)
///  - collect macros after the preamble of the main file (in ParsedAST.cpp)
class CollectMainFileMacros : public PPCallbacks {
public:
  explicit CollectMainFileMacros(const Preprocessor &PP, MainFileMacros &Out)
      : SM(PP.getSourceManager()), PP(PP), Out(Out) {}

  void FileChanged(SourceLocation Loc, FileChangeReason,
                   SrcMgr::CharacteristicKind, FileID) override;

  void MacroDefined(const Token &MacroName, const MacroDirective *MD) override;

  void MacroExpands(const Token &MacroName, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;

  void MacroUndefined(const clang::Token &MacroName,
                      const clang::MacroDefinition &MD,
                      const clang::MacroDirective *Undef) override;

  void Ifdef(SourceLocation Loc, const Token &MacroName,
             const MacroDefinition &MD) override;
  void Ifndef(SourceLocation Loc, const Token &MacroName,
              const MacroDefinition &MD) override;
  using PPCallbacks::Elifdef;
  using PPCallbacks::Elifndef;
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override;
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override;

  void Defined(const Token &MacroName, const MacroDefinition &MD,
               SourceRange Range) override;

  void SourceRangeSkipped(SourceRange R, SourceLocation EndifLoc) override;

  // Called when the AST build is done to disable further recording
  // of macros by this class. This is needed because some clang-tidy
  // checks can trigger PP callbacks by calling directly into the
  // preprocessor. Such calls are not interleaved with FileChanged()
  // in the expected way, leading this class to erroneously process
  // macros that are not in the main file.
  void doneParse() { InMainFile = false; }

private:
  void add(const Token &MacroNameTok, const MacroInfo *MI,
           bool IsDefinition = false, bool InConditionalDirective = false);
  const SourceManager &SM;
  const Preprocessor &PP;
  bool InMainFile = true;
  MainFileMacros &Out;
};

/// Represents a `#pragma mark` in the main file.
///
/// There can be at most one pragma mark per line.
struct PragmaMark {
  Range Rng;
  std::string Trivia;
};

/// Collect all pragma marks from the main file.
std::unique_ptr<PPCallbacks>
collectPragmaMarksCallback(const SourceManager &, std::vector<PragmaMark> &Out);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_COLLECTMACROS_H
