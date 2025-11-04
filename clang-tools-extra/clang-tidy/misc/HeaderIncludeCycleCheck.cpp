//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderIncludeCycleCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Regex.h"
#include <algorithm>
#include <optional>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

struct Include {
  const FileEntry *File;
  StringRef Name;
  SourceLocation Loc;
};

class CyclicDependencyCallbacks : public PPCallbacks {
public:
  CyclicDependencyCallbacks(HeaderIncludeCycleCheck &Check,
                            const SourceManager &SM,
                            const std::vector<StringRef> &IgnoredFilesList)
      : Check(Check), SM(SM) {
    IgnoredFilesRegexes.reserve(IgnoredFilesList.size());
    for (const StringRef &It : IgnoredFilesList) {
      if (!It.empty())
        IgnoredFilesRegexes.emplace_back(It);
    }
  }

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    if (FileType != clang::SrcMgr::C_User)
      return;

    if (Reason != EnterFile && Reason != ExitFile)
      return;

    const FileID Id = SM.getFileID(Loc);
    if (Id.isInvalid())
      return;

    const FileEntry *NewFile = SM.getFileEntryForID(Id);
    const FileEntry *PrevFile = SM.getFileEntryForID(PrevFID);

    if (Reason == ExitFile) {
      if ((Files.size() > 1U) && (Files.back().File == PrevFile) &&
          (Files[Files.size() - 2U].File == NewFile))
        Files.pop_back();
      return;
    }

    if (!Files.empty() && Files.back().File == NewFile)
      return;

    const std::optional<StringRef> FilePath = SM.getNonBuiltinFilenameForID(Id);
    const StringRef FileName =
        FilePath ? llvm::sys::path::filename(*FilePath) : StringRef();
    Files.push_back({NewFile, FileName, std::exchange(NextToEnter, {})});
  }

  void InclusionDirective(SourceLocation, const Token &, StringRef FilePath,
                          bool, CharSourceRange Range,
                          OptionalFileEntryRef File, StringRef, StringRef,
                          const Module *, bool,
                          SrcMgr::CharacteristicKind FileType) override {
    if (FileType != clang::SrcMgr::C_User)
      return;

    NextToEnter = Range.getBegin();

    if (!File)
      return;

    checkForDoubleInclude(&File->getFileEntry(),
                          llvm::sys::path::filename(FilePath),
                          Range.getBegin());
  }

  void checkForDoubleInclude(const FileEntry *File, StringRef FileName,
                             SourceLocation Loc) {
    const auto It =
        llvm::find_if(llvm::reverse(Files),
                      [&](const Include &Entry) { return Entry.File == File; });
    if (It == Files.rend())
      return;

    const StringRef FilePath = File->tryGetRealPathName();
    if (FilePath.empty() || isFileIgnored(FilePath))
      return;

    if (It == Files.rbegin()) {
      Check.diag(Loc, "direct self-inclusion of header file '%0'") << FileName;
      return;
    }

    Check.diag(Loc, "circular header file dependency detected while including "
                    "'%0', please check the include path")
        << FileName;

    const bool IsIncludePathValid =
        std::all_of(Files.rbegin(), It + 1, [](const Include &Elem) {
          return !Elem.Name.empty() && Elem.Loc.isValid();
        });
    if (!IsIncludePathValid)
      return;

    for (const Include &I : llvm::make_range(Files.rbegin(), It + 1))
      Check.diag(I.Loc, "'%0' included from here", DiagnosticIDs::Note)
          << I.Name;
  }

  bool isFileIgnored(StringRef FileName) const {
    return llvm::any_of(IgnoredFilesRegexes, [&](const llvm::Regex &It) {
      return It.match(FileName);
    });
  }

#ifndef NDEBUG
  void EndOfMainFile() override {
    if (!Files.empty() &&
        Files.back().File == SM.getFileEntryForID(SM.getMainFileID()))
      Files.pop_back();

    assert(Files.empty());
  }
#endif

private:
  std::vector<Include> Files;
  SourceLocation NextToEnter;
  HeaderIncludeCycleCheck &Check;
  const SourceManager &SM;
  std::vector<llvm::Regex> IgnoredFilesRegexes;
};

} // namespace

HeaderIncludeCycleCheck::HeaderIncludeCycleCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoredFilesList(utils::options::parseStringList(
          Options.get("IgnoredFilesList", ""))) {}

void HeaderIncludeCycleCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      std::make_unique<CyclicDependencyCallbacks>(*this, SM, IgnoredFilesList));
}

void HeaderIncludeCycleCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredFilesList",
                utils::options::serializeStringList(IgnoredFilesList));
}

} // namespace clang::tidy::misc
