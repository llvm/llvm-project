//===--- Rewriter.h - Code rewriting interface ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Rewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITER_H
#define LLVM_CLANG_REWRITER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Rewrite/Core/DeltaTree.h"
#include "clang/Rewrite/Core/RewriteRope.h"
#include "llvm/ADT/StringRef.h"
#include <cstring>
#include <map>
#include <string>

namespace clang {
  class LangOptions;
  class Rewriter;
  class SourceManager;
  class Stmt;

/// RewriteBuffer - As code is rewritten, SourceBuffer's from the original
/// input with modifications get a new RewriteBuffer associated with them.  The
/// RewriteBuffer captures the modified text itself as well as information used
/// to map between SourceLocation's in the original input and offsets in the
/// RewriteBuffer.  For example, if text is inserted into the buffer, any
/// locations after the insertion point have to be mapped.
class RewriteBuffer {
  friend class Rewriter;
  /// Deltas - Keep track of all the deltas in the source code due to insertions
  /// and deletions.
  DeltaTree Deltas;

  /// Buffer - This is the actual buffer itself.  Note that using a vector or
  /// string is a horribly inefficient way to do this, we should use a rope
  /// instead.
  typedef RewriteRope BufferTy;
  BufferTy Buffer;
public:
  typedef BufferTy::const_iterator iterator;
  iterator begin() const { return Buffer.begin(); }
  iterator end() const { return Buffer.end(); }
  unsigned size() const { return Buffer.size(); }

  raw_ostream &write(raw_ostream &) const;

  /// RemoveText - Remove the specified text.
  void RemoveText(unsigned OrigOffset, unsigned Size,
                  bool removeLineIfEmpty = false);

  /// InsertText - Insert some text at the specified point, where the offset in
  /// the buffer is specified relative to the original SourceBuffer.  The
  /// text is inserted after the specified location.
  ///
  void InsertText(unsigned OrigOffset, StringRef Str,
                  bool InsertAfter = true);


  /// InsertTextBefore - Insert some text before the specified point, where the
  /// offset in the buffer is specified relative to the original
  /// SourceBuffer. The text is inserted before the specified location.  This is
  /// method is the same as InsertText with "InsertAfter == false".
  void InsertTextBefore(unsigned OrigOffset, StringRef Str) {
    InsertText(OrigOffset, Str, false);
  }

  /// InsertTextAfter - Insert some text at the specified point, where the
  /// offset in the buffer is specified relative to the original SourceBuffer.
  /// The text is inserted after the specified location.
  void InsertTextAfter(unsigned OrigOffset, StringRef Str) {
    InsertText(OrigOffset, Str);
  }

  /// ReplaceText - This method replaces a range of characters in the input
  /// buffer with a new string.  This is effectively a combined "remove/insert"
  /// operation.
  void ReplaceText(unsigned OrigOffset, unsigned OrigLength,
                   StringRef NewStr);

private:  // Methods only usable by Rewriter.

  /// Initialize - Start this rewrite buffer out with a copy of the unmodified
  /// input buffer.
  void Initialize(const char *BufStart, const char *BufEnd) {
    Buffer.assign(BufStart, BufEnd);
  }

  /// getMappedOffset - Given an offset into the original SourceBuffer that this
  /// RewriteBuffer is based on, map it into the offset space of the
  /// RewriteBuffer.  If AfterInserts is true and if the OrigOffset indicates a
  /// position where text is inserted, the location returned will be after any
  /// inserted text at the position.
  unsigned getMappedOffset(unsigned OrigOffset,
                           bool AfterInserts = false) const{
    return Deltas.getDeltaAt(2*OrigOffset+AfterInserts)+OrigOffset;
  }

  /// AddInsertDelta - When an insertion is made at a position, this
  /// method is used to record that information.
  void AddInsertDelta(unsigned OrigOffset, int Change) {
    return Deltas.AddDelta(2*OrigOffset, Change);
  }

  /// AddReplaceDelta - When a replacement/deletion is made at a position, this
  /// method is used to record that information.
  void AddReplaceDelta(unsigned OrigOffset, int Change) {
    return Deltas.AddDelta(2*OrigOffset+1, Change);
  }
};


/// Rewriter - This is the main interface to the rewrite buffers.  Its primary
/// job is to dispatch high-level requests to the low-level RewriteBuffers that
/// are involved.
class Rewriter {
  SourceManager *SourceMgr;
  const LangOptions *LangOpts;
  std::map<FileID, RewriteBuffer> RewriteBuffers;
public:
  struct RewriteOptions {
    /// \brief Given a source range, true to include previous inserts at the
    /// beginning of the range as part of the range itself (true by default).
    bool IncludeInsertsAtBeginOfRange;
    /// \brief Given a source range, true to include previous inserts at the
    /// end of the range as part of the range itself (true by default).
    bool IncludeInsertsAtEndOfRange;
    /// \brief If true and removing some text leaves a blank line
    /// also remove the empty line (false by default).
    bool RemoveLineIfEmpty;

    RewriteOptions()
      : IncludeInsertsAtBeginOfRange(true),
        IncludeInsertsAtEndOfRange(true),
        RemoveLineIfEmpty(false) { }
  };

  typedef std::map<FileID, RewriteBuffer>::iterator buffer_iterator;

  explicit Rewriter(SourceManager &SM, const LangOptions &LO)
    : SourceMgr(&SM), LangOpts(&LO) {}
  explicit Rewriter() : SourceMgr(0), LangOpts(0) {}

  void setSourceMgr(SourceManager &SM, const LangOptions &LO) {
    SourceMgr = &SM;
    LangOpts = &LO;
  }
  SourceManager &getSourceMgr() const { return *SourceMgr; }
  const LangOptions &getLangOpts() const { return *LangOpts; }

  /// isRewritable - Return true if this location is a raw file location, which
  /// is rewritable.  Locations from macros, etc are not rewritable.
  static bool isRewritable(SourceLocation Loc) {
    return Loc.isFileID();
  }

  /// getRangeSize - Return the size in bytes of the specified range if they
  /// are in the same file.  If not, this returns -1.
  int getRangeSize(SourceRange Range,
                   RewriteOptions opts = RewriteOptions()) const;
  int getRangeSize(const CharSourceRange &Range,
                   RewriteOptions opts = RewriteOptions()) const;

  /// getRewrittenText - Return the rewritten form of the text in the specified
  /// range.  If the start or end of the range was unrewritable or if they are
  /// in different buffers, this returns an empty string.
  ///
  /// Note that this method is not particularly efficient.
  ///
  std::string getRewrittenText(SourceRange Range) const;

  /// InsertText - Insert the specified string at the specified location in the
  /// original buffer.  This method returns true (and does nothing) if the input
  /// location was not rewritable, false otherwise.
  ///
  /// \param indentNewLines if true new lines in the string are indented
  /// using the indentation of the source line in position \p Loc.
  bool InsertText(SourceLocation Loc, StringRef Str,
                  bool InsertAfter = true, bool indentNewLines = false);

  /// InsertTextAfter - Insert the specified string at the specified location in
  ///  the original buffer.  This method returns true (and does nothing) if
  ///  the input location was not rewritable, false otherwise.  Text is
  ///  inserted after any other text that has been previously inserted
  ///  at the some point (the default behavior for InsertText).
  bool InsertTextAfter(SourceLocation Loc, StringRef Str) {
    return InsertText(Loc, Str);
  }

  /// \brief Insert the specified string after the token in the
  /// specified location.
  bool InsertTextAfterToken(SourceLocation Loc, StringRef Str);

  /// InsertText - Insert the specified string at the specified location in the
  /// original buffer.  This method returns true (and does nothing) if the input
  /// location was not rewritable, false otherwise.  Text is
  /// inserted before any other text that has been previously inserted
  /// at the some point.
  bool InsertTextBefore(SourceLocation Loc, StringRef Str) {
    return InsertText(Loc, Str, false);
  }

  /// RemoveText - Remove the specified text region.
  bool RemoveText(SourceLocation Start, unsigned Length,
                  RewriteOptions opts = RewriteOptions());

  /// \brief Remove the specified text region.
  bool RemoveText(CharSourceRange range,
                  RewriteOptions opts = RewriteOptions()) {
    return RemoveText(range.getBegin(), getRangeSize(range, opts), opts);
  }

  /// \brief Remove the specified text region.
  bool RemoveText(SourceRange range, RewriteOptions opts = RewriteOptions()) {
    return RemoveText(range.getBegin(), getRangeSize(range, opts), opts);
  }

  /// ReplaceText - This method replaces a range of characters in the input
  /// buffer with a new string.  This is effectively a combined "remove/insert"
  /// operation.
  bool ReplaceText(SourceLocation Start, unsigned OrigLength,
                   StringRef NewStr);

  /// ReplaceText - This method replaces a range of characters in the input
  /// buffer with a new string.  This is effectively a combined "remove/insert"
  /// operation.
  bool ReplaceText(SourceRange range, StringRef NewStr) {
    return ReplaceText(range.getBegin(), getRangeSize(range), NewStr);
  }

  /// ReplaceText - This method replaces a range of characters in the input
  /// buffer with a new string.  This is effectively a combined "remove/insert"
  /// operation.
  bool ReplaceText(SourceRange range, SourceRange replacementRange);

  /// ReplaceStmt - This replaces a Stmt/Expr with another, using the pretty
  /// printer to generate the replacement code.  This returns true if the input
  /// could not be rewritten, or false if successful.
  bool ReplaceStmt(Stmt *From, Stmt *To);

  /// \brief Increase indentation for the lines between the given source range.
  /// To determine what the indentation should be, 'parentIndent' is used
  /// that should be at a source location with an indentation one degree
  /// lower than the given range.
  bool IncreaseIndentation(CharSourceRange range, SourceLocation parentIndent);
  bool IncreaseIndentation(SourceRange range, SourceLocation parentIndent) {
    return IncreaseIndentation(CharSourceRange::getTokenRange(range),
                               parentIndent);
  }

  /// ConvertToString converts statement 'From' to a string using the
  /// pretty printer.
  std::string ConvertToString(Stmt *From);

  /// getEditBuffer - This is like getRewriteBufferFor, but always returns a
  /// buffer, and allows you to write on it directly.  This is useful if you
  /// want efficient low-level access to apis for scribbling on one specific
  /// FileID's buffer.
  RewriteBuffer &getEditBuffer(FileID FID);

  /// getRewriteBufferFor - Return the rewrite buffer for the specified FileID.
  /// If no modification has been made to it, return null.
  const RewriteBuffer *getRewriteBufferFor(FileID FID) const {
    std::map<FileID, RewriteBuffer>::const_iterator I =
      RewriteBuffers.find(FID);
    return I == RewriteBuffers.end() ? 0 : &I->second;
  }

  // Iterators over rewrite buffers.
  buffer_iterator buffer_begin() { return RewriteBuffers.begin(); }
  buffer_iterator buffer_end() { return RewriteBuffers.end(); }

  /// SaveFiles - Save all changed files to disk.
  ///
  /// Returns whether not all changes were saved successfully.
  /// Outputs diagnostics via the source manager's diagnostic engine
  /// in case of an error.
  bool overwriteChangedFiles();

private:
  unsigned getLocationOffsetAndFileID(SourceLocation Loc, FileID &FID) const;
};

} // end namespace clang

#endif
