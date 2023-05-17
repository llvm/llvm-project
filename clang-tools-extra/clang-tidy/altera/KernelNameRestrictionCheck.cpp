//===--- KernelNameRestrictionCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelNameRestrictionCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include <string>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::altera {

namespace {

class KernelNameRestrictionPPCallbacks : public PPCallbacks {
public:
  explicit KernelNameRestrictionPPCallbacks(ClangTidyCheck &Check,
                                            const SourceManager &SM)
      : Check(Check), SM(SM) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FileNameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

  void EndOfMainFile() override;

private:
  /// Returns true if the name of the file with path FilePath is 'kernel.cl',
  /// 'verilog.cl', or 'vhdl.cl'. The file name check is case insensitive.
  bool fileNameIsRestricted(StringRef FilePath);

  struct IncludeDirective {
    SourceLocation Loc; // Location in the include directive.
    StringRef FileName; // Filename as a string.
  };

  std::vector<IncludeDirective> IncludeDirectives;
  ClangTidyCheck &Check;
  const SourceManager &SM;
};

} // namespace

void KernelNameRestrictionCheck::registerPPCallbacks(const SourceManager &SM,
                                                     Preprocessor *PP,
                                                     Preprocessor *) {
  PP->addPPCallbacks(
      std::make_unique<KernelNameRestrictionPPCallbacks>(*this, SM));
}

void KernelNameRestrictionPPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &, StringRef FileName, bool,
    CharSourceRange, OptionalFileEntryRef, StringRef, StringRef, const Module *,
    SrcMgr::CharacteristicKind) {
  IncludeDirective ID = {HashLoc, FileName};
  IncludeDirectives.push_back(std::move(ID));
}

bool KernelNameRestrictionPPCallbacks::fileNameIsRestricted(
    StringRef FileName) {
  return FileName.equals_insensitive("kernel.cl") ||
         FileName.equals_insensitive("verilog.cl") ||
         FileName.equals_insensitive("vhdl.cl");
}

void KernelNameRestrictionPPCallbacks::EndOfMainFile() {

  // Check main file for restricted names.
  const FileEntry *Entry = SM.getFileEntryForID(SM.getMainFileID());
  StringRef FileName = llvm::sys::path::filename(Entry->getName());
  if (fileNameIsRestricted(FileName))
    Check.diag(SM.getLocForStartOfFile(SM.getMainFileID()),
               "compiling '%0' may cause additional compilation errors due "
               "to the name of the kernel source file; consider renaming the "
               "included kernel source file")
        << FileName;

  if (IncludeDirectives.empty())
    return;

  // Check included files for restricted names.
  for (const IncludeDirective &ID : IncludeDirectives) {
    StringRef FileName = llvm::sys::path::filename(ID.FileName);
    if (fileNameIsRestricted(FileName))
      Check.diag(ID.Loc,
                 "including '%0' may cause additional compilation errors due "
                 "to the name of the kernel source file; consider renaming the "
                 "included kernel source file")
          << FileName;
  }
}

} // namespace clang::tidy::altera
