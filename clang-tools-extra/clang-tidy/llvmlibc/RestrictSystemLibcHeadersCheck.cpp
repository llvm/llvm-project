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

namespace clang {
namespace tidy {
namespace llvm_libc {

namespace {

class RestrictedIncludesPPCallbacks : public PPCallbacks {
public:
  explicit RestrictedIncludesPPCallbacks(
      RestrictSystemLibcHeadersCheck &Check, const SourceManager &SM,
      const SmallString<128> CompilerIncudeDir)
      : Check(Check), SM(SM), CompilerIncudeDir(CompilerIncudeDir) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

private:
  RestrictSystemLibcHeadersCheck &Check;
  const SourceManager &SM;
  const SmallString<128> CompilerIncudeDir;
};

} // namespace

void RestrictedIncludesPPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (SrcMgr::isSystem(FileType)) {
    // Compiler provided headers are allowed (e.g stddef.h).
    if (SearchPath == CompilerIncudeDir) return;
    if (!SM.isInMainFile(HashLoc)) {
      Check.diag(
          HashLoc,
          "system libc header %0 not allowed, transitively included from %1")
          << FileName << SM.getFilename(HashLoc);
    } else {
      Check.diag(HashLoc, "system libc header %0 not allowed") << FileName;
    }
  }
}

void RestrictSystemLibcHeadersCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  SmallString<128> CompilerIncudeDir =
      StringRef(PP->getHeaderSearchInfo().getHeaderSearchOpts().ResourceDir);
  llvm::sys::path::append(CompilerIncudeDir, "include");
  PP->addPPCallbacks(std::make_unique<RestrictedIncludesPPCallbacks>(
      *this, SM, CompilerIncudeDir));
}

} // namespace llvm_libc
} // namespace tidy
} // namespace clang
