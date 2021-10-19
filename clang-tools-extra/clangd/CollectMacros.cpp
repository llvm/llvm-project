//===--- CollectMacros.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CollectMacros.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace clangd {

void CollectMainFileMacros::add(const Token &MacroNameTok, const MacroInfo *MI,
                                bool IsDefinition) {
  if (!InMainFile)
    return;
  auto Loc = MacroNameTok.getLocation();
  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  auto Name = MacroNameTok.getIdentifierInfo()->getName();
  Out.Names.insert(Name);
  auto Range = halfOpenToRange(
      SM, CharSourceRange::getCharRange(Loc, MacroNameTok.getEndLoc()));
  if (auto SID = getSymbolID(Name, MI, SM))
    Out.MacroRefs[SID].push_back({Range, IsDefinition});
  else
    Out.UnknownMacros.push_back({Range, IsDefinition});
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

} // namespace clangd
} // namespace clang
