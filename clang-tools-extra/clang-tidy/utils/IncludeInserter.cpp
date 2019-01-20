//===-------- IncludeInserter.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeInserter.h"
#include "clang/Lex/Token.h"

namespace clang {
namespace tidy {
namespace utils {

class IncludeInserterCallback : public PPCallbacks {
public:
  explicit IncludeInserterCallback(IncludeInserter *Inserter)
      : Inserter(Inserter) {}
  // Implements PPCallbacks::InclusionDerective(). Records the names and source
  // locations of the inclusions in the main source file being processed.
  void InclusionDirective(SourceLocation HashLocation,
                          const Token &IncludeToken, StringRef FileNameRef,
                          bool IsAngled, CharSourceRange FileNameRange,
                          const FileEntry * /*IncludedFile*/,
                          StringRef /*SearchPath*/, StringRef /*RelativePath*/,
                          const Module * /*ImportedModule*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    Inserter->AddInclude(FileNameRef, IsAngled, HashLocation,
                         IncludeToken.getEndLoc());
  }

private:
  IncludeInserter *Inserter;
};

IncludeInserter::IncludeInserter(const SourceManager &SourceMgr,
                                 const LangOptions &LangOpts,
                                 IncludeSorter::IncludeStyle Style)
    : SourceMgr(SourceMgr), LangOpts(LangOpts), Style(Style) {}

IncludeInserter::~IncludeInserter() {}

std::unique_ptr<PPCallbacks> IncludeInserter::CreatePPCallbacks() {
  return llvm::make_unique<IncludeInserterCallback>(this);
}

llvm::Optional<FixItHint>
IncludeInserter::CreateIncludeInsertion(FileID FileID, StringRef Header,
                                        bool IsAngled) {
  // We assume the same Header will never be included both angled and not
  // angled.
  if (!InsertedHeaders[FileID].insert(Header).second)
    return llvm::None;

  if (IncludeSorterByFile.find(FileID) == IncludeSorterByFile.end()) {
    // This may happen if there have been no preprocessor directives in this
    // file.
    IncludeSorterByFile.insert(std::make_pair(
        FileID,
        llvm::make_unique<IncludeSorter>(
            &SourceMgr, &LangOpts, FileID,
            SourceMgr.getFilename(SourceMgr.getLocForStartOfFile(FileID)),
            Style)));
  }
  return IncludeSorterByFile[FileID]->CreateIncludeInsertion(Header, IsAngled);
}

void IncludeInserter::AddInclude(StringRef FileName, bool IsAngled,
                                 SourceLocation HashLocation,
                                 SourceLocation EndLocation) {
  FileID FileID = SourceMgr.getFileID(HashLocation);
  if (IncludeSorterByFile.find(FileID) == IncludeSorterByFile.end()) {
    IncludeSorterByFile.insert(std::make_pair(
        FileID, llvm::make_unique<IncludeSorter>(
                    &SourceMgr, &LangOpts, FileID,
                    SourceMgr.getFilename(HashLocation), Style)));
  }
  IncludeSorterByFile[FileID]->AddInclude(FileName, IsAngled, HashLocation,
                                          EndLocation);
}

} // namespace utils
} // namespace tidy
} // namespace clang
