//===--- RestrictSystemIncludesCheck.h - clang-tidy --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_RESTRICTINCLUDESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_RESTRICTINCLUDESSCHECK_H

#include "../ClangTidyCheck.h"
#include "../GlobList.h"
#include "clang/Lex/PPCallbacks.h"

namespace clang::tidy::portability {

/// Checks for allowed includes and suggests removal of any others. If no
/// includes are specified, the check will exit without issuing any warnings.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability/restrict-system-includes.html
class RestrictSystemIncludesCheck : public ClangTidyCheck {
public:
  RestrictSystemIncludesCheck(StringRef Name, ClangTidyContext *Context,
                              std::string DefaultAllowedIncludes = "*")
      : ClangTidyCheck(Name, Context),
        AllowedIncludes(Options.get("Includes", DefaultAllowedIncludes)),
        AllowedIncludesGlobList(AllowedIncludes) {}

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool contains(StringRef FileName) {
    return AllowedIncludesGlobList.contains(FileName);
  }

private:
  std::string AllowedIncludes;
  GlobList AllowedIncludesGlobList;
};

class RestrictedIncludesPPCallbacks : public PPCallbacks {
public:
  explicit RestrictedIncludesPPCallbacks(RestrictSystemIncludesCheck &Check,
                                         const SourceManager &SM)
      : Check(Check), SM(SM) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
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
  const SourceManager &SM;
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_RESTRICTINCLUDESSCHECK_H
