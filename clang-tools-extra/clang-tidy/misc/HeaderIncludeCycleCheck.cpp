//===--- HeaderIncludeCycleCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderIncludeCycleCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Regex.h"
#include <algorithm>
#include <deque>
#include <optional>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

struct Include {
  FileID Id;
  llvm::StringRef Name;
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

    FileID Id = SM.getFileID(Loc);
    if (Id.isInvalid())
      return;

    if (Reason == ExitFile) {
      if ((Files.size() > 1U) && (Files.back().Id == PrevFID) &&
          (Files[Files.size() - 2U].Id == Id))
        Files.pop_back();
      return;
    }

    if (!Files.empty() && Files.back().Id == Id)
      return;

    std::optional<llvm::StringRef> FilePath = SM.getNonBuiltinFilenameForID(Id);
    llvm::StringRef FileName =
        FilePath ? llvm::sys::path::filename(*FilePath) : llvm::StringRef();

    if (!NextToEnter)
      NextToEnter = Include{Id, FileName, SourceLocation()};

    assert(NextToEnter->Name == FileName);
    NextToEnter->Id = Id;
    Files.emplace_back(*NextToEnter);
    NextToEnter.reset();
  }

  void InclusionDirective(SourceLocation, const Token &, StringRef FilePath,
                          bool, CharSourceRange Range,
                          OptionalFileEntryRef File, StringRef, StringRef,
                          const Module *, bool,
                          SrcMgr::CharacteristicKind FileType) override {
    if (FileType != clang::SrcMgr::C_User)
      return;

    llvm::StringRef FileName = llvm::sys::path::filename(FilePath);
    NextToEnter = {FileID(), FileName, Range.getBegin()};

    if (!File)
      return;

    FileID Id = SM.translateFile(*File);
    if (Id.isInvalid())
      return;

    checkForDoubleInclude(Id, FileName, Range.getBegin());
  }

  void EndOfMainFile() override {
    if (!Files.empty() && Files.back().Id == SM.getMainFileID())
      Files.pop_back();

    assert(Files.empty());
  }

  void checkForDoubleInclude(FileID Id, llvm::StringRef FileName,
                             SourceLocation Loc) {
    auto It =
        std::find_if(Files.rbegin(), Files.rend(),
                     [&](const Include &Entry) { return Entry.Id == Id; });
    if (It == Files.rend())
      return;

    const std::optional<StringRef> FilePath = SM.getNonBuiltinFilenameForID(Id);
    if (!FilePath || isFileIgnored(*FilePath))
      return;

    if (It == Files.rbegin()) {
      Check.diag(Loc, "direct self-inclusion of header file '%0'") << FileName;
      return;
    }

    Check.diag(Loc, "circular header file dependency detected while including "
                    "'%0', please check the include path")
        << FileName;

    const bool IsIncludePathValid =
        std::all_of(Files.rbegin(), It, [](const Include &Elem) {
          return !Elem.Name.empty() && Elem.Loc.isValid();
        });

    if (!IsIncludePathValid)
      return;

    auto CurrentIt = Files.rbegin();
    do {
      Check.diag(CurrentIt->Loc, "'%0' included from here", DiagnosticIDs::Note)
          << CurrentIt->Name;
    } while (CurrentIt++ != It);
  }

  bool isFileIgnored(StringRef FileName) const {
    return llvm::any_of(IgnoredFilesRegexes, [&](const llvm::Regex &It) {
      return It.match(FileName);
    });
  }

private:
  std::deque<Include> Files;
  std::optional<Include> NextToEnter;
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
