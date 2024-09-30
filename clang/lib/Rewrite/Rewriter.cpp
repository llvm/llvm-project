//===- Rewriter.cpp - Code rewriting interface ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Rewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/RewriteBuffer.h"
#include "llvm/ADT/RewriteRope.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <map>
#include <utility>

using namespace clang;
using llvm::RewriteBuffer;

//===----------------------------------------------------------------------===//
// Rewriter class
//===----------------------------------------------------------------------===//

/// Return true if this character is non-new-line whitespace:
/// ' ', '\\t', '\\f', '\\v', '\\r'.
static inline bool isWhitespaceExceptNL(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\f' || c == '\v' || c == '\r';
}

/// getRangeSize - Return the size in bytes of the specified range if they
/// are in the same file.  If not, this returns -1.
int Rewriter::getRangeSize(const CharSourceRange &Range,
                           RewriteOptions opts) const {
  if (!isRewritable(Range.getBegin()) ||
      !isRewritable(Range.getEnd())) return -1;

  FileID StartFileID, EndFileID;
  unsigned StartOff = getLocationOffsetAndFileID(Range.getBegin(), StartFileID);
  unsigned EndOff = getLocationOffsetAndFileID(Range.getEnd(), EndFileID);

  if (StartFileID != EndFileID)
    return -1;

  // If edits have been made to this buffer, the delta between the range may
  // have changed.
  std::map<FileID, RewriteBuffer>::const_iterator I =
    RewriteBuffers.find(StartFileID);
  if (I != RewriteBuffers.end()) {
    const RewriteBuffer &RB = I->second;
    EndOff = RB.getMappedOffset(EndOff, opts.IncludeInsertsAtEndOfRange);
    StartOff = RB.getMappedOffset(StartOff, !opts.IncludeInsertsAtBeginOfRange);
  }

  // Adjust the end offset to the end of the last token, instead of being the
  // start of the last token if this is a token range.
  if (Range.isTokenRange())
    EndOff += Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr, *LangOpts);

  return EndOff-StartOff;
}

int Rewriter::getRangeSize(SourceRange Range, RewriteOptions opts) const {
  return getRangeSize(CharSourceRange::getTokenRange(Range), opts);
}

/// getRewrittenText - Return the rewritten form of the text in the specified
/// range.  If the start or end of the range was unrewritable or if they are
/// in different buffers, this returns an empty string.
///
/// Note that this method is not particularly efficient.
std::string Rewriter::getRewrittenText(CharSourceRange Range) const {
  if (!isRewritable(Range.getBegin()) ||
      !isRewritable(Range.getEnd()))
    return {};

  FileID StartFileID, EndFileID;
  unsigned StartOff, EndOff;
  StartOff = getLocationOffsetAndFileID(Range.getBegin(), StartFileID);
  EndOff   = getLocationOffsetAndFileID(Range.getEnd(), EndFileID);

  if (StartFileID != EndFileID)
    return {}; // Start and end in different buffers.

  // If edits have been made to this buffer, the delta between the range may
  // have changed.
  std::map<FileID, RewriteBuffer>::const_iterator I =
    RewriteBuffers.find(StartFileID);
  if (I == RewriteBuffers.end()) {
    // If the buffer hasn't been rewritten, just return the text from the input.
    const char *Ptr = SourceMgr->getCharacterData(Range.getBegin());

    // Adjust the end offset to the end of the last token, instead of being the
    // start of the last token.
    if (Range.isTokenRange())
      EndOff +=
          Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr, *LangOpts);
    return std::string(Ptr, Ptr+EndOff-StartOff);
  }

  const RewriteBuffer &RB = I->second;
  EndOff = RB.getMappedOffset(EndOff, true);
  StartOff = RB.getMappedOffset(StartOff);

  // Adjust the end offset to the end of the last token, instead of being the
  // start of the last token.
  if (Range.isTokenRange())
    EndOff += Lexer::MeasureTokenLength(Range.getEnd(), *SourceMgr, *LangOpts);

  // Advance the iterators to the right spot, yay for linear time algorithms.
  RewriteBuffer::iterator Start = RB.begin();
  std::advance(Start, StartOff);
  RewriteBuffer::iterator End = Start;
  assert(EndOff >= StartOff && "Invalid iteration distance");
  std::advance(End, EndOff-StartOff);

  return std::string(Start, End);
}

unsigned Rewriter::getLocationOffsetAndFileID(SourceLocation Loc,
                                              FileID &FID) const {
  assert(Loc.isValid() && "Invalid location");
  std::pair<FileID, unsigned> V = SourceMgr->getDecomposedLoc(Loc);
  FID = V.first;
  return V.second;
}

/// getEditBuffer - Get or create a RewriteBuffer for the specified FileID.
RewriteBuffer &Rewriter::getEditBuffer(FileID FID) {
  std::map<FileID, RewriteBuffer>::iterator I =
    RewriteBuffers.lower_bound(FID);
  if (I != RewriteBuffers.end() && I->first == FID)
    return I->second;
  I = RewriteBuffers.insert(I, std::make_pair(FID, RewriteBuffer()));

  StringRef MB = SourceMgr->getBufferData(FID);
  I->second.Initialize(MB.begin(), MB.end());

  return I->second;
}

/// InsertText - Insert the specified string at the specified location in the
/// original buffer.
bool Rewriter::InsertText(SourceLocation Loc, StringRef Str,
                          bool InsertAfter, bool indentNewLines) {
  if (!isRewritable(Loc)) return true;
  FileID FID;
  unsigned StartOffs = getLocationOffsetAndFileID(Loc, FID);

  SmallString<128> indentedStr;
  if (indentNewLines && Str.contains('\n')) {
    StringRef MB = SourceMgr->getBufferData(FID);

    unsigned lineNo = SourceMgr->getLineNumber(FID, StartOffs) - 1;
    const SrcMgr::ContentCache *Content =
        &SourceMgr->getSLocEntry(FID).getFile().getContentCache();
    unsigned lineOffs = Content->SourceLineCache[lineNo];

    // Find the whitespace at the start of the line.
    StringRef indentSpace;
    {
      unsigned i = lineOffs;
      while (isWhitespaceExceptNL(MB[i]))
        ++i;
      indentSpace = MB.substr(lineOffs, i-lineOffs);
    }

    SmallVector<StringRef, 4> lines;
    Str.split(lines, "\n");

    for (unsigned i = 0, e = lines.size(); i != e; ++i) {
      indentedStr += lines[i];
      if (i < e-1) {
        indentedStr += '\n';
        indentedStr += indentSpace;
      }
    }
    Str = indentedStr.str();
  }

  getEditBuffer(FID).InsertText(StartOffs, Str, InsertAfter);
  return false;
}

bool Rewriter::InsertTextAfterToken(SourceLocation Loc, StringRef Str) {
  if (!isRewritable(Loc)) return true;
  FileID FID;
  unsigned StartOffs = getLocationOffsetAndFileID(Loc, FID);
  RewriteOptions rangeOpts;
  rangeOpts.IncludeInsertsAtBeginOfRange = false;
  StartOffs += getRangeSize(SourceRange(Loc, Loc), rangeOpts);
  getEditBuffer(FID).InsertText(StartOffs, Str, /*InsertAfter*/true);
  return false;
}

/// RemoveText - Remove the specified text region.
bool Rewriter::RemoveText(SourceLocation Start, unsigned Length,
                          RewriteOptions opts) {
  if (!isRewritable(Start)) return true;
  FileID FID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, FID);
  getEditBuffer(FID).RemoveText(StartOffs, Length, opts.RemoveLineIfEmpty);
  return false;
}

/// ReplaceText - This method replaces a range of characters in the input
/// buffer with a new string.  This is effectively a combined "remove/insert"
/// operation.
bool Rewriter::ReplaceText(SourceLocation Start, unsigned OrigLength,
                           StringRef NewStr) {
  if (!isRewritable(Start)) return true;
  FileID StartFileID;
  unsigned StartOffs = getLocationOffsetAndFileID(Start, StartFileID);

  getEditBuffer(StartFileID).ReplaceText(StartOffs, OrigLength, NewStr);
  return false;
}

bool Rewriter::ReplaceText(SourceRange range, SourceRange replacementRange) {
  if (!isRewritable(range.getBegin())) return true;
  if (!isRewritable(range.getEnd())) return true;
  if (replacementRange.isInvalid()) return true;
  SourceLocation start = range.getBegin();
  unsigned origLength = getRangeSize(range);
  unsigned newLength = getRangeSize(replacementRange);
  FileID FID;
  unsigned newOffs = getLocationOffsetAndFileID(replacementRange.getBegin(),
                                                FID);
  StringRef MB = SourceMgr->getBufferData(FID);
  return ReplaceText(start, origLength, MB.substr(newOffs, newLength));
}

bool Rewriter::IncreaseIndentation(CharSourceRange range,
                                   SourceLocation parentIndent) {
  if (range.isInvalid()) return true;
  if (!isRewritable(range.getBegin())) return true;
  if (!isRewritable(range.getEnd())) return true;
  if (!isRewritable(parentIndent)) return true;

  FileID StartFileID, EndFileID, parentFileID;
  unsigned StartOff, EndOff, parentOff;

  StartOff = getLocationOffsetAndFileID(range.getBegin(), StartFileID);
  EndOff   = getLocationOffsetAndFileID(range.getEnd(), EndFileID);
  parentOff = getLocationOffsetAndFileID(parentIndent, parentFileID);

  if (StartFileID != EndFileID || StartFileID != parentFileID)
    return true;
  if (StartOff > EndOff)
    return true;

  FileID FID = StartFileID;
  StringRef MB = SourceMgr->getBufferData(FID);

  unsigned parentLineNo = SourceMgr->getLineNumber(FID, parentOff) - 1;
  unsigned startLineNo = SourceMgr->getLineNumber(FID, StartOff) - 1;
  unsigned endLineNo = SourceMgr->getLineNumber(FID, EndOff) - 1;

  const SrcMgr::ContentCache *Content =
      &SourceMgr->getSLocEntry(FID).getFile().getContentCache();

  // Find where the lines start.
  unsigned parentLineOffs = Content->SourceLineCache[parentLineNo];
  unsigned startLineOffs = Content->SourceLineCache[startLineNo];

  // Find the whitespace at the start of each line.
  StringRef parentSpace, startSpace;
  {
    unsigned i = parentLineOffs;
    while (isWhitespaceExceptNL(MB[i]))
      ++i;
    parentSpace = MB.substr(parentLineOffs, i-parentLineOffs);

    i = startLineOffs;
    while (isWhitespaceExceptNL(MB[i]))
      ++i;
    startSpace = MB.substr(startLineOffs, i-startLineOffs);
  }
  if (parentSpace.size() >= startSpace.size())
    return true;
  if (!startSpace.starts_with(parentSpace))
    return true;

  StringRef indent = startSpace.substr(parentSpace.size());

  // Indent the lines between start/end offsets.
  RewriteBuffer &RB = getEditBuffer(FID);
  for (unsigned lineNo = startLineNo; lineNo <= endLineNo; ++lineNo) {
    unsigned offs = Content->SourceLineCache[lineNo];
    unsigned i = offs;
    while (isWhitespaceExceptNL(MB[i]))
      ++i;
    StringRef origIndent = MB.substr(offs, i-offs);
    if (origIndent.starts_with(startSpace))
      RB.InsertText(offs, indent, /*InsertAfter=*/false);
  }

  return false;
}

bool Rewriter::overwriteChangedFiles() {
  bool AllWritten = true;
  auto& Diag = getSourceMgr().getDiagnostics();
  unsigned OverwriteFailure = Diag.getCustomDiagID(
      DiagnosticsEngine::Error, "unable to overwrite file %0: %1");
  for (buffer_iterator I = buffer_begin(), E = buffer_end(); I != E; ++I) {
    OptionalFileEntryRef Entry = getSourceMgr().getFileEntryRefForID(I->first);
    llvm::SmallString<128> Path(Entry->getName());
    getSourceMgr().getFileManager().makeAbsolutePath(Path);
    if (auto Error = llvm::writeToOutput(Path, [&](llvm::raw_ostream &OS) {
          I->second.write(OS);
          return llvm::Error::success();
        })) {
      Diag.Report(OverwriteFailure)
          << Entry->getName() << llvm::toString(std::move(Error));
      AllWritten = false;
    }
  }
  return !AllWritten;
}
