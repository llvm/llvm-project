//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DuplicateIncludeCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <memory>
#include <vector>

namespace clang::tidy::readability {

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
                            const SourceManager &SM,
                            const std::vector<std::string> &AllowedStrings);

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
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
  std::vector<llvm::Regex> AllowedDuplicateRegex;

  bool IsAllowedDuplicateInclude(StringRef TokenName, OptionalFileEntryRef File,
                                 StringRef RelativePath);
};

} // namespace

DuplicateIncludeCallbacks::DuplicateIncludeCallbacks(
    DuplicateIncludeCheck &Check, const SourceManager &SM,
    const std::vector<std::string> &AllowedStrings)
    : Check(Check), SM(SM) {
  // The main file doesn't participate in the FileChanged notification.
  Files.emplace_back();
  AllowedDuplicateRegex.reserve(AllowedStrings.size());
  for (const std::string &str : AllowedStrings) {
    AllowedDuplicateRegex.emplace_back(str);
  }
}

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
    bool IsAngled, CharSourceRange FilenameRange, OptionalFileEntryRef File,
    StringRef SearchPath, StringRef RelativePath, const Module *SuggestedModule,
    bool ModuleImported, SrcMgr::CharacteristicKind FileType) {
  // Skip includes behind macros
  if (FilenameRange.getBegin().isMacroID() ||
      FilenameRange.getEnd().isMacroID())
    return;

  // if duplicate allowed, record and return
  if (IsAllowedDuplicateInclude(FileName, File, RelativePath)) {
    Files.back().push_back(FileName);
    return;
  }

  if (llvm::is_contained(Files.back(), FileName)) {
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

bool DuplicateIncludeCallbacks::IsAllowedDuplicateInclude(
    StringRef TokenName, OptionalFileEntryRef File, StringRef RelativePath) {
  SmallVector<StringRef, 3> matchArguments;
  matchArguments.push_back(TokenName);

  if (!RelativePath.empty())
    matchArguments.push_back(llvm::sys::path::filename(RelativePath));

  if (File) {
    StringRef RealPath = File->getFileEntry().tryGetRealPathName();
    if (!RealPath.empty())
      matchArguments.push_back(llvm::sys::path::filename(RealPath));
  }

  // try to match with each regex
  for (const llvm::Regex &reg : AllowedDuplicateRegex) {
    for (StringRef arg : matchArguments) {
      if (reg.match(arg))
        return true;
    }
  }
  return false;
}

DuplicateIncludeCheck::DuplicateIncludeCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  std::string Raw = Options.get("AllowedDuplicateIncludes", "").str();
  if (!Raw.empty()) {
    SmallVector<StringRef, 4> StringParts;
    StringRef(Raw).split(StringParts, ',', -1, false);

    for (StringRef Part : StringParts) {
      Part = Part.trim();
      if (!Part.empty())
        AllowedDuplicateIncludes.push_back(Part.str());
    }
  }
}

void DuplicateIncludeCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<DuplicateIncludeCallbacks>(
      *this, SM, AllowedDuplicateIncludes));
}

void DuplicateIncludeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedDuplicateIncludes",
                llvm::join(AllowedDuplicateIncludes, ","));
}
} // namespace clang::tidy::readability
