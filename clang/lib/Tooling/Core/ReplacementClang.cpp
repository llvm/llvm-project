//===- ReplacementNoClang.cpp - Framework for clang refactoring tools -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements classes to support/store refactorings. This file contains all
//  of the code that depends on Clang, so that Replacement.cpp can be used from
//  tools used to build Clang, like tblgen.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/RewriteBuffer.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace tooling;

const LangOptions Replacement::DefaultLangOptions;

Replacement::Replacement(const SourceManager &Sources, SourceLocation Start,
                         unsigned Length, StringRef ReplacementText) {
  setFromSourceLocation(Sources, Start, Length, ReplacementText);
}

Replacement::Replacement(const SourceManager &Sources,
                         const CharSourceRange &Range,
                         StringRef ReplacementText,
                         const LangOptions &LangOpts) {
  setFromSourceRange(Sources, Range, ReplacementText, LangOpts);
}

bool Replacement::apply(Rewriter &Rewrite) const {
  SourceManager &SM = Rewrite.getSourceMgr();
  auto Entry = SM.getFileManager().getOptionalFileRef(FilePath);
  if (!Entry)
    return false;

  FileID ID = SM.getOrCreateFileID(*Entry, SrcMgr::C_User);
  const SourceLocation Start = SM.getLocForStartOfFile(ID).getLocWithOffset(
      ReplacementRange.getOffset());
  // ReplaceText returns false on success.
  // ReplaceText only fails if the source location is not a file location, in
  // which case we already returned false earlier.
  bool RewriteSucceeded = !Rewrite.ReplaceText(
      Start, ReplacementRange.getLength(), ReplacementText);
  assert(RewriteSucceeded);
  return RewriteSucceeded;
}

void Replacement::setFromSourceLocation(const SourceManager &Sources,
                                        SourceLocation Start, unsigned Length,
                                        StringRef ReplacementText) {
  const FileIDAndOffset DecomposedLocation = Sources.getDecomposedLoc(Start);
  OptionalFileEntryRef Entry =
      Sources.getFileEntryRefForID(DecomposedLocation.first);
  this->FilePath = std::string(Entry ? Entry->getName() : InvalidLocation);
  this->ReplacementRange = Range(DecomposedLocation.second, Length);
  this->ReplacementText = std::string(ReplacementText);
}

// FIXME: This should go into the Lexer, but we need to figure out how
// to handle ranges for refactoring in general first - there is no obvious
// good way how to integrate this into the Lexer yet.
static int getRangeSize(const SourceManager &Sources,
                        const CharSourceRange &Range,
                        const LangOptions &LangOpts) {
  SourceLocation SpellingBegin = Sources.getSpellingLoc(Range.getBegin());
  SourceLocation SpellingEnd = Sources.getSpellingLoc(Range.getEnd());
  FileIDAndOffset Start = Sources.getDecomposedLoc(SpellingBegin);
  FileIDAndOffset End = Sources.getDecomposedLoc(SpellingEnd);
  if (Start.first != End.first)
    return -1;
  if (Range.isTokenRange())
    End.second += Lexer::MeasureTokenLength(SpellingEnd, Sources, LangOpts);
  return End.second - Start.second;
}

void Replacement::setFromSourceRange(const SourceManager &Sources,
                                     const CharSourceRange &Range,
                                     StringRef ReplacementText,
                                     const LangOptions &LangOpts) {
  setFromSourceLocation(Sources, Sources.getSpellingLoc(Range.getBegin()),
                        getRangeSize(Sources, Range, LangOpts),
                        ReplacementText);
}

void Replacement::setFromSourceRange(const SourceManager &Sources,
                                     const SourceRange &Range,
                                     StringRef ReplacementText,
                                     const LangOptions &LangOpts) {
  setFromSourceRange(Sources, CharSourceRange::getTokenRange(Range),
                     ReplacementText, LangOpts);
}

namespace clang {
namespace tooling {

bool applyAllReplacements(const Replacements &Replaces, Rewriter &Rewrite) {
  bool Result = true;
  for (auto I = Replaces.rbegin(), E = Replaces.rend(); I != E; ++I) {
    if (I->isApplicable()) {
      Result = I->apply(Rewrite) && Result;
    } else {
      Result = false;
    }
  }
  return Result;
}

llvm::Expected<std::string> applyAllReplacements(StringRef Code,
                                                 const Replacements &Replaces) {
  if (Replaces.empty())
    return Code.str();

  auto InMemoryFileSystem =
      llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager Files(FileSystemOptions(), InMemoryFileSystem);
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diagnostics(DiagnosticIDs::create(), DiagOpts);
  SourceManager SourceMgr(Diagnostics, Files);
  Rewriter Rewrite(SourceMgr, LangOptions());
  InMemoryFileSystem->addFile(
      "<stdin>", 0, llvm::MemoryBuffer::getMemBuffer(Code, "<stdin>"));
  FileID ID = SourceMgr.createFileID(*Files.getOptionalFileRef("<stdin>"),
                                     SourceLocation(), clang::SrcMgr::C_User);
  for (auto I = Replaces.rbegin(), E = Replaces.rend(); I != E; ++I) {
    Replacement Replace("<stdin>", I->getOffset(), I->getLength(),
                        I->getReplacementText());
    if (!Replace.apply(Rewrite))
      return llvm::make_error<ReplacementError>(
          replacement_error::fail_to_apply, Replace);
  }
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  Rewrite.getEditBuffer(ID).write(OS);
  return Result;
}

std::map<std::string, Replacements> groupReplacementsByFile(
    FileManager &FileMgr,
    const std::map<std::string, Replacements> &FileToReplaces) {
  std::map<std::string, Replacements> Result;
  llvm::SmallPtrSet<const FileEntry *, 16> ProcessedFileEntries;
  for (const auto &Entry : FileToReplaces) {
    auto FE = FileMgr.getOptionalFileRef(Entry.first);
    if (!FE)
      llvm::errs() << "File path " << Entry.first << " is invalid.\n";
    else if (ProcessedFileEntries.insert(*FE).second)
      Result[Entry.first] = std::move(Entry.second);
  }
  return Result;
}

} // namespace tooling
} // namespace clang
