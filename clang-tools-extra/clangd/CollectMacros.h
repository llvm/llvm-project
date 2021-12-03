//===--- CollectMacros.h -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COLLECTMACROS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COLLECTMACROS_H

#include "AST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "index/SymbolID.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace clang {
namespace clangd {

struct MacroOccurrence {
  // Instead of storing SourceLocation, we have to store the token range because
  // SourceManager from preamble is not available when we build the AST.
  Range Rng;
  bool IsDefinition;
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
  explicit CollectMainFileMacros(const SourceManager &SM, MainFileMacros &Out)
      : SM(SM), Out(Out) {}

  void FileChanged(SourceLocation Loc, FileChangeReason,
                   SrcMgr::CharacteristicKind, FileID) override {
    InMainFile = isInsideMainFile(Loc, SM);
  }

  void MacroDefined(const Token &MacroName, const MacroDirective *MD) override {
    add(MacroName, MD->getMacroInfo(), /*IsDefinition=*/true);
  }

  void MacroExpands(const Token &MacroName, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    add(MacroName, MD.getMacroInfo());
  }

  void MacroUndefined(const clang::Token &MacroName,
                      const clang::MacroDefinition &MD,
                      const clang::MacroDirective *Undef) override {
    add(MacroName, MD.getMacroInfo());
  }

  void Ifdef(SourceLocation Loc, const Token &MacroName,
             const MacroDefinition &MD) override {
    add(MacroName, MD.getMacroInfo());
  }

  void Ifndef(SourceLocation Loc, const Token &MacroName,
              const MacroDefinition &MD) override {
    add(MacroName, MD.getMacroInfo());
  }

  void Defined(const Token &MacroName, const MacroDefinition &MD,
               SourceRange Range) override {
    add(MacroName, MD.getMacroInfo());
  }

  void SourceRangeSkipped(SourceRange R, SourceLocation EndifLoc) override {
    if (!InMainFile)
      return;
    Position Begin = sourceLocToPosition(SM, R.getBegin());
    Position End = sourceLocToPosition(SM, R.getEnd());
    Out.SkippedRanges.push_back(Range{Begin, End});
  }

private:
  void add(const Token &MacroNameTok, const MacroInfo *MI,
           bool IsDefinition = false);
  const SourceManager &SM;
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
