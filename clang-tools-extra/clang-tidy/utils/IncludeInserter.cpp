//===----------------------------------------------------------------------===//
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
                          const Token &IncludeToken, StringRef FileName,
                          bool IsAngled, CharSourceRange FileNameRange,
                          OptionalFileEntryRef /*IncludedFile*/,
                          StringRef /*SearchPath*/, StringRef /*RelativePath*/,
                          const Module * /*SuggestedModule*/,
                          bool /*ModuleImported*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    const FileID FileID = Inserter->SourceMgr->getFileID(HashLocation);
    auto &[InsertionLoc, AlreadyPresentHeaders] = Inserter->InsertInfos[FileID];
    AlreadyPresentHeaders.try_emplace(FileName);
    if (InsertionLoc.isInvalid())
      InsertionLoc = HashLocation;
  }

private:
  IncludeInserter *Inserter;
};

IncludeInserter::IncludeInserter(bool SelfContainedDiags)
    : SelfContainedDiags(SelfContainedDiags) {}

void IncludeInserter::registerPreprocessor(Preprocessor *PP) {
  assert(PP && "PP shouldn't be null");
  SourceMgr = &PP->getSourceManager();

  // If this gets registered multiple times, clear the map.
  InsertInfos.clear();
  PP->addPPCallbacks(std::make_unique<IncludeInserterCallback>(this));
}

std::optional<FixItHint>
IncludeInserter::createIncludeInsertion(FileID FileID, StringRef Header) {
  const bool IsAngled = Header.consume_front("<");
  if (IsAngled != Header.consume_back(">"))
    return std::nullopt;

  auto &[InsertionLoc, AlreadyPresentHeaders] = InsertInfos[FileID];

  // We assume the same Header will never be included both angled and not
  // angled.
  // In self contained diags mode we don't track what headers we have already
  // inserted.
  if (!SelfContainedDiags && !AlreadyPresentHeaders.try_emplace(Header).second)
    return std::nullopt;

  std::string IncludeStmt = IsAngled
                                ? Twine("#include <" + Header + ">\n").str()
                                : Twine("#include \"" + Header + "\"\n").str();

  // If there are no includes in this file, add it in the first line.
  // FIXME: insert after the file comment or the header guard, if present.
  if (InsertionLoc.isInvalid()) {
    InsertionLoc = SourceMgr->getLocForStartOfFile(FileID);
    IncludeStmt += '\n';
  }

  return FixItHint::CreateInsertion(InsertionLoc, IncludeStmt);
}

std::optional<FixItHint>
IncludeInserter::createMainFileIncludeInsertion(StringRef Header) {
  assert(SourceMgr && "SourceMgr shouldn't be null; did you remember to call "
                      "registerPreprocessor()?");
  return createIncludeInsertion(SourceMgr->getMainFileID(), Header);
}

} // namespace clang::tidy::utils
