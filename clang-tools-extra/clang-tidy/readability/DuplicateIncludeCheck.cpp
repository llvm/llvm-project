//===--- DuplicateIncludeCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DuplicateIncludeCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace clang {
namespace tidy {
namespace readability {

static SourceLocation advanceBeyondCurrentLine(const SourceManager &SM,
                                               SourceLocation Start,
                                               int Offset) {
  const FileID Id = SM.getFileID(Start);
  const unsigned LineNumber = SM.getSpellingLineNumber(Start);
  while (SM.getFileID(Start) == Id &&
         SM.getSpellingLineNumber(Start.getLocWithOffset(Offset)) == LineNumber)
    Start = Start.getLocWithOffset(Offset);
  return Start;
}

namespace {

using FileList = SmallVector<StringRef>;

class DuplicateIncludeCallbacks : public PPCallbacks {
public:
  DuplicateIncludeCallbacks(DuplicateIncludeCheck &Check,
                            const SourceManager &SM)
      : Check(Check), SM(SM) {
    // The main file doesn't participate in the FileChanged notification.
    Files.emplace_back();
  }

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          Optional<FileEntryRef> File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;

  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override;

private:
  // A list of included files is kept for each file we enter.
  SmallVector<FileList> Files;
  DuplicateIncludeCheck &Check;
  const SourceManager &SM;
};

void DuplicateIncludeCallbacks::FileChanged(SourceLocation Loc,
                                            FileChangeReason Reason,
                                            SrcMgr::CharacteristicKind FileType,
                                            FileID PrevFID) {
  if (Reason == EnterFile)
    Files.emplace_back();
  else if (Reason == ExitFile)
    Files.pop_back();
}

void DuplicateIncludeCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, Optional<FileEntryRef> File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (llvm::find(Files.back(), FileName) != Files.back().end()) {
    // We want to delete the entire line, so make sure that [Start,End] covers
    // everything.
    SourceLocation Start =
        advanceBeyondCurrentLine(SM, HashLoc, -1).getLocWithOffset(-1);
    SourceLocation End =
        advanceBeyondCurrentLine(SM, FilenameRange.getEnd(), 1);
    Check.diag(HashLoc, "duplicate include")
        << FixItHint::CreateRemoval(SourceRange{Start, End});
  } else
    Files.back().push_back(FileName);
}

void DuplicateIncludeCallbacks::MacroDefined(const Token &MacroNameTok,
                                             const MacroDirective *MD) {
  Files.back().clear();
}

void DuplicateIncludeCallbacks::MacroUndefined(const Token &MacroNameTok,
                                               const MacroDefinition &MD,
                                               const MacroDirective *Undef) {
  Files.back().clear();
}

} // namespace

void DuplicateIncludeCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<DuplicateIncludeCallbacks>(*this, SM));
}

} // namespace readability
} // namespace tidy
} // namespace clang
