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
    : SourceMgr(SourceMgr), Style(Style) {}

IncludeInserter::~IncludeInserter() {}

std::unique_ptr<PPCallbacks> IncludeInserter::CreatePPCallbacks() {
  return std::make_unique<IncludeInserterCallback>(this);
}

IncludeSorter &IncludeInserter::getOrCreate(FileID FileID) {
  // std::unique_ptr is cheap to construct, so force a construction now to save
  // the lookup needed if we were to insert into the map.
  std::unique_ptr<IncludeSorter> &Entry = IncludeSorterByFile[FileID];
  if (!Entry) {
    // If it wasn't found, Entry will be default constructed to nullptr.
    Entry = std::make_unique<IncludeSorter>(
        &SourceMgr, FileID,
        SourceMgr.getFilename(SourceMgr.getLocForStartOfFile(FileID)), Style);
  }
  return *Entry;
}

llvm::Optional<FixItHint>
IncludeInserter::CreateIncludeInsertion(FileID FileID, StringRef Header,
                                        bool IsAngled) {
  // We assume the same Header will never be included both angled and not
  // angled.
  if (!InsertedHeaders[FileID].insert(std::string(Header)).second)
    return llvm::None;

  return getOrCreate(FileID).CreateIncludeInsertion(Header, IsAngled);
}

void IncludeInserter::AddInclude(StringRef FileName, bool IsAngled,
                                 SourceLocation HashLocation,
                                 SourceLocation EndLocation) {
  FileID FileID = SourceMgr.getFileID(HashLocation);
  getOrCreate(FileID).AddInclude(FileName, IsAngled, HashLocation, EndLocation);
}

} // namespace utils
} // namespace tidy
} // namespace clang
