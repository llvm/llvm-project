//===--- RestrictSystemIncludesCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

namespace clang::tidy::portability {

void RestrictedIncludesPPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, OptionalFileEntryRef File,
    StringRef SearchPath, StringRef RelativePath, const Module *SuggestedModule,
    bool ModuleImported, SrcMgr::CharacteristicKind FileType) {
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
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      std::make_unique<RestrictedIncludesPPCallbacks>(*this, SM));
}

void RestrictSystemIncludesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Includes", AllowedIncludes);
}

} // namespace clang::tidy::portability
