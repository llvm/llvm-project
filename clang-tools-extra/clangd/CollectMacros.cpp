//===--- CollectMacros.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CollectMacros.h"
#include "AST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/STLExtras.h"
#include <cstddef>

namespace clang {
namespace clangd {

Range MacroOccurrence::toRange(const SourceManager &SM) const {
  auto MainFile = SM.getMainFileID();
  return halfOpenToRange(
      SM, syntax::FileRange(MainFile, StartOffset, EndOffset).toCharRange(SM));
}

void CollectMainFileMacros::add(const Token &MacroNameTok, const MacroInfo *MI,
                                bool IsDefinition, bool InIfCondition) {
  if (!InMainFile)
    return;
  auto Loc = MacroNameTok.getLocation();
  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  auto Name = MacroNameTok.getIdentifierInfo()->getName();
  Out.Names.insert(Name);
  size_t Start = SM.getFileOffset(Loc);
  size_t End = SM.getFileOffset(MacroNameTok.getEndLoc());
  if (auto SID = getSymbolID(Name, MI, SM))
    Out.MacroRefs[SID].push_back({Start, End, IsDefinition, InIfCondition});
  else
    Out.UnknownMacros.push_back({Start, End, IsDefinition, InIfCondition});
}

void CollectMainFileMacros::FileChanged(SourceLocation Loc, FileChangeReason,
                                        SrcMgr::CharacteristicKind, FileID) {
  InMainFile = isInsideMainFile(Loc, SM);
}

void CollectMainFileMacros::MacroExpands(const Token &MacroName,
                                         const MacroDefinition &MD,
                                         SourceRange Range,
                                         const MacroArgs *Args) {
  add(MacroName, MD.getMacroInfo());
}

void CollectMainFileMacros::MacroUndefined(const clang::Token &MacroName,
                                           const clang::MacroDefinition &MD,
                                           const clang::MacroDirective *Undef) {
  add(MacroName, MD.getMacroInfo());
}

void CollectMainFileMacros::Ifdef(SourceLocation Loc, const Token &MacroName,
                                  const MacroDefinition &MD) {
  add(MacroName, MD.getMacroInfo(), /*IsDefinition=*/false,
      /*InConditionalDirective=*/true);
}

void CollectMainFileMacros::Ifndef(SourceLocation Loc, const Token &MacroName,
                                   const MacroDefinition &MD) {
  add(MacroName, MD.getMacroInfo(), /*IsDefinition=*/false,
      /*InConditionalDirective=*/true);
}

void CollectMainFileMacros::Elifdef(SourceLocation Loc, const Token &MacroName,
                                    const MacroDefinition &MD) {
  add(MacroName, MD.getMacroInfo(), /*IsDefinition=*/false,
      /*InConditionalDirective=*/true);
}

void CollectMainFileMacros::Elifndef(SourceLocation Loc, const Token &MacroName,
                                     const MacroDefinition &MD) {
  add(MacroName, MD.getMacroInfo(), /*IsDefinition=*/false,
      /*InConditionalDirective=*/true);
}

void CollectMainFileMacros::Defined(const Token &MacroName,
                                    const MacroDefinition &MD,
                                    SourceRange Range) {
  add(MacroName, MD.getMacroInfo(), /*IsDefinition=*/false,
      /*InConditionalDirective=*/true);
}

void CollectMainFileMacros::SourceRangeSkipped(SourceRange R,
                                               SourceLocation EndifLoc) {
  if (!InMainFile)
    return;
  Position Begin = sourceLocToPosition(SM, R.getBegin());
  Position End = sourceLocToPosition(SM, R.getEnd());
  Out.SkippedRanges.push_back(Range{Begin, End});
}

class CollectPragmaMarks : public PPCallbacks {
public:
  explicit CollectPragmaMarks(const SourceManager &SM,
                              std::vector<clangd::PragmaMark> &Out)
      : SM(SM), Out(Out) {}

  void PragmaMark(SourceLocation Loc, StringRef Trivia) override {
    if (isInsideMainFile(Loc, SM)) {
      // FIXME: This range should just cover `XX` in `#pragma mark XX` and
      // `- XX` in `#pragma mark - XX`.
      Position Start = sourceLocToPosition(SM, Loc);
      Position End = {Start.line + 1, 0};
      Out.emplace_back(clangd::PragmaMark{{Start, End}, Trivia.str()});
    }
  }

private:
  const SourceManager &SM;
  std::vector<clangd::PragmaMark> &Out;
};

std::unique_ptr<PPCallbacks>
collectPragmaMarksCallback(const SourceManager &SM,
                           std::vector<PragmaMark> &Out) {
  return std::make_unique<CollectPragmaMarks>(SM, Out);
}

void CollectMainFileMacros::MacroDefined(const Token &MacroName,
                                         const MacroDirective *MD) {

  if (!InMainFile)
    return;
  const auto *MI = MD->getMacroInfo();
  add(MacroName, MD->getMacroInfo(), true);
  if (MI)
    for (const auto &Tok : MI->tokens()) {
      auto *II = Tok.getIdentifierInfo();
      // Could this token be a reference to a macro? (Not param to this macro).
      if (!II || !II->hadMacroDefinition() ||
          llvm::is_contained(MI->params(), II))
        continue;
      if (const MacroInfo *MI = PP.getMacroInfo(II))
        add(Tok, MI);
    }
}

} // namespace clangd
} // namespace clang
