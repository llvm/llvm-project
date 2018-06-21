//===--- RestrictSystemIncludesCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RestrictSystemIncludesCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include <cstring>

namespace clang {
namespace tidy {
namespace fuchsia {

class RestrictedIncludesPPCallbacks : public PPCallbacks {
public:
  explicit RestrictedIncludesPPCallbacks(RestrictSystemIncludesCheck &Check,
                                         SourceManager &SM)
      : Check(Check), SM(SM) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;
  void EndOfMainFile() override;

private:
  struct IncludeDirective {
    IncludeDirective() = default;
    IncludeDirective(SourceLocation Loc, CharSourceRange Range,
                     StringRef Filename, StringRef FullPath, bool IsInMainFile)
        : Loc(Loc), Range(Range), IncludeFile(Filename), IncludePath(FullPath),
          IsInMainFile(IsInMainFile) {}

    SourceLocation Loc;      // '#' location in the include directive
    CharSourceRange Range;   // SourceRange for the file name
    std::string IncludeFile; // Filename as a string
    std::string IncludePath; // Full file path as a string
    bool IsInMainFile;       // Whether or not the include is in the main file
  };

  using FileIncludes = llvm::SmallVector<IncludeDirective, 8>;
  llvm::SmallDenseMap<FileID, FileIncludes> IncludeDirectives;

  RestrictSystemIncludesCheck &Check;
  SourceManager &SM;
};

void RestrictedIncludesPPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (!Check.contains(FileName) && SrcMgr::isSystem(FileType)) {
    SmallString<256> FullPath;
    llvm::sys::path::append(FullPath, SearchPath);
    llvm::sys::path::append(FullPath, RelativePath);
    // Bucket the allowed include directives by the id of the file they were
    // declared in.
    IncludeDirectives[SM.getFileID(HashLoc)].emplace_back(
        HashLoc, FilenameRange, FileName, FullPath.str(),
        SM.isInMainFile(HashLoc));
  }
}

void RestrictedIncludesPPCallbacks::EndOfMainFile() {
  for (const auto &Bucket : IncludeDirectives) {
    const FileIncludes &FileDirectives = Bucket.second;

    // Emit fixits for all restricted includes.
    for (const auto &Include : FileDirectives) {
      // Fetch the length of the include statement from the start to just after
      // the newline, for finding the end (including the newline).
      unsigned ToLen = std::strcspn(SM.getCharacterData(Include.Loc), "\n") + 1;
      CharSourceRange ToRange = CharSourceRange::getCharRange(
          Include.Loc, Include.Loc.getLocWithOffset(ToLen));

      if (!Include.IsInMainFile) {
        auto D = Check.diag(
            Include.Loc,
            "system include %0 not allowed, transitively included from %1");
        D << Include.IncludeFile << SM.getFilename(Include.Loc);
        D << FixItHint::CreateRemoval(ToRange);
        continue;
      }
      auto D = Check.diag(Include.Loc, "system include %0 not allowed");
      D << Include.IncludeFile;
      D << FixItHint::CreateRemoval(ToRange);
    }
  }
}

void RestrictSystemIncludesCheck::registerPPCallbacks(
    CompilerInstance &Compiler) {
  Compiler.getPreprocessor().addPPCallbacks(
      llvm::make_unique<RestrictedIncludesPPCallbacks>(
          *this, Compiler.getSourceManager()));
}

void RestrictSystemIncludesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Includes", AllowedIncludes);
}

} // namespace fuchsia
} // namespace tidy
} // namespace clang
