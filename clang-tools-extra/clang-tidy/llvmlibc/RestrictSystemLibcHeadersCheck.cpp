//===--- RestrictSystemLibcHeadersCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RestrictSystemLibcHeadersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"

// FixItHint - Hint to check documentation script to mark this check as
// providing a FixIt.

namespace clang::tidy::llvm_libc {

namespace {

class RestrictedIncludesPPCallbacks
    : public portability::RestrictedIncludesPPCallbacks {
public:
  explicit RestrictedIncludesPPCallbacks(
      RestrictSystemLibcHeadersCheck &Check, const SourceManager &SM,
      const SmallString<128> CompilerIncudeDir)
      : portability::RestrictedIncludesPPCallbacks(Check, SM),
        CompilerIncudeDir(CompilerIncudeDir) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override;

private:
  const SmallString<128> CompilerIncudeDir;
};

} // namespace

void RestrictedIncludesPPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, OptionalFileEntryRef File,
    StringRef SearchPath, StringRef RelativePath, const Module *SuggestedModule,
    bool ModuleImported, SrcMgr::CharacteristicKind FileType) {
  // Compiler provided headers are allowed (e.g stddef.h).
  if (SrcMgr::isSystem(FileType) && SearchPath == CompilerIncudeDir)
    return;
  portability::RestrictedIncludesPPCallbacks::InclusionDirective(
      HashLoc, IncludeTok, FileName, IsAngled, FilenameRange, File, SearchPath,
      RelativePath, SuggestedModule, ModuleImported, FileType);
}

void RestrictSystemLibcHeadersCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  SmallString<128> CompilerIncudeDir =
      StringRef(PP->getHeaderSearchInfo().getHeaderSearchOpts().ResourceDir);
  llvm::sys::path::append(CompilerIncudeDir, "include");
  PP->addPPCallbacks(std::make_unique<RestrictedIncludesPPCallbacks>(
      *this, SM, CompilerIncudeDir));
}

} // namespace clang::tidy::llvm_libc
