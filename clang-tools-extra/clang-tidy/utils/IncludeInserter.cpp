//===-------- IncludeInserter.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeInserter.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include <optional>

namespace clang::tidy::utils {

class IncludeInserterCallback : public PPCallbacks {
public:
  explicit IncludeInserterCallback(IncludeInserter *Inserter)
      : Inserter(Inserter) {}
  // Implements PPCallbacks::InclusionDirective(). Records the names and source
  // locations of the inclusions in the main source file being processed.
  void InclusionDirective(SourceLocation HashLocation,
                          const Token &IncludeToken, StringRef FileNameRef,
                          bool IsAngled, CharSourceRange FileNameRange,
                          OptionalFileEntryRef /*IncludedFile*/,
                          StringRef /*SearchPath*/, StringRef /*RelativePath*/,
                          const Module * /*ImportedModule*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    Inserter->addInclude(FileNameRef, IsAngled, HashLocation,
                         IncludeToken.getEndLoc());
  }

private:
  IncludeInserter *Inserter;
};

IncludeInserter::IncludeInserter(IncludeSorter::IncludeStyle Style,
                                 bool SelfContainedDiags)
    : Style(Style), SelfContainedDiags(SelfContainedDiags) {}

void IncludeInserter::registerPreprocessor(Preprocessor *PP) {
  assert(PP && "PP shouldn't be null");
  SourceMgr = &PP->getSourceManager();

  // If this gets registered multiple times, clear the maps
  if (!IncludeSorterByFile.empty())
    IncludeSorterByFile.clear();
  if (!InsertedHeaders.empty())
    InsertedHeaders.clear();
  PP->addPPCallbacks(std::make_unique<IncludeInserterCallback>(this));
}

IncludeSorter &IncludeInserter::getOrCreate(FileID FileID) {
  assert(SourceMgr && "SourceMgr shouldn't be null; did you remember to call "
                      "registerPreprocessor()?");
  // std::unique_ptr is cheap to construct, so force a construction now to save
  // the lookup needed if we were to insert into the map.
  std::unique_ptr<IncludeSorter> &Entry = IncludeSorterByFile[FileID];
  if (!Entry) {
    // If it wasn't found, Entry will be default constructed to nullptr.
    Entry = std::make_unique<IncludeSorter>(
        SourceMgr, FileID,
        SourceMgr->getFilename(SourceMgr->getLocForStartOfFile(FileID)), Style);
  }
  return *Entry;
}

std::optional<FixItHint>
IncludeInserter::createIncludeInsertion(FileID FileID, llvm::StringRef Header) {
  bool IsAngled = Header.consume_front("<");
  if (IsAngled != Header.consume_back(">"))
    return std::nullopt;
  // We assume the same Header will never be included both angled and not
  // angled.
  // In self contained diags mode we don't track what headers we have already
  // inserted.
  if (!SelfContainedDiags && !InsertedHeaders[FileID].insert(Header).second)
    return std::nullopt;

  return getOrCreate(FileID).createIncludeInsertion(Header, IsAngled);
}

std::optional<FixItHint>
IncludeInserter::createMainFileIncludeInsertion(StringRef Header) {
  assert(SourceMgr && "SourceMgr shouldn't be null; did you remember to call "
                      "registerPreprocessor()?");
  return createIncludeInsertion(SourceMgr->getMainFileID(), Header);
}

void IncludeInserter::addInclude(StringRef FileName, bool IsAngled,
                                 SourceLocation HashLocation,
                                 SourceLocation EndLocation) {
  assert(SourceMgr && "SourceMgr shouldn't be null; did you remember to call "
                      "registerPreprocessor()?");
  FileID FileID = SourceMgr->getFileID(HashLocation);
  getOrCreate(FileID).addInclude(FileName, IsAngled, HashLocation, EndLocation);
}

} // namespace clang::tidy::utils
