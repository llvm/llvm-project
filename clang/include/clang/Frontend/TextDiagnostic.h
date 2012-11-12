//===--- TextDiagnostic.h - Text Diagnostic Pretty-Printing -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class that provides support for textual pretty-printing of
// diagnostics. It is used to implement the different code paths which require
// such functionality in a consistent way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_H_
#define LLVM_CLANG_FRONTEND_TEXT_DIAGNOSTIC_H_

#include "clang/Frontend/DiagnosticRenderer.h"

struct SourceColumnMap;

namespace clang {

/// \brief Class to encapsulate the logic for formatting and printing a textual
/// diagnostic message.
///
/// This class provides an interface for building and emitting a textual
/// diagnostic, including all of the macro backtraces, caret diagnostics, FixIt
/// Hints, and code snippets. In the presence of macros this involves
/// a recursive process, synthesizing notes for each macro expansion.
///
/// The purpose of this class is to isolate the implementation of printing
/// beautiful text diagnostics from any particular interfaces. The Clang
/// DiagnosticClient is implemented through this class as is diagnostic
/// printing coming out of libclang.
class TextDiagnostic : public DiagnosticRenderer {
  raw_ostream &OS;

public:
  TextDiagnostic(raw_ostream &OS,
                 const LangOptions &LangOpts,
                 DiagnosticOptions *DiagOpts);

  virtual ~TextDiagnostic();
  
  /// \brief Print the diagonstic level to a raw_ostream.
  ///
  /// This is a static helper that handles colorizing the level and formatting
  /// it into an arbitrary output stream. This is used internally by the
  /// TextDiagnostic emission code, but it can also be used directly by
  /// consumers that don't have a source manager or other state that the full
  /// TextDiagnostic logic requires.
  static void printDiagnosticLevel(raw_ostream &OS,
                                   DiagnosticsEngine::Level Level,
                                   bool ShowColors);

  /// \brief Pretty-print a diagnostic message to a raw_ostream.
  ///
  /// This is a static helper to handle the line wrapping, colorizing, and
  /// rendering of a diagnostic message to a particular ostream. It is
  /// publicly visible so that clients which do not have sufficient state to
  /// build a complete TextDiagnostic object can still get consistent
  /// formatting of their diagnostic messages.
  ///
  /// \param OS Where the message is printed
  /// \param Level Used to colorizing the message
  /// \param Message The text actually printed
  /// \param CurrentColumn The starting column of the first line, accounting
  ///                      for any prefix.
  /// \param Columns The number of columns to use in line-wrapping, 0 disables
  ///                all line-wrapping.
  /// \param ShowColors Enable colorizing of the message.
  static void printDiagnosticMessage(raw_ostream &OS,
                                     DiagnosticsEngine::Level Level,
                                     StringRef Message,
                                     unsigned CurrentColumn, unsigned Columns,
                                     bool ShowColors);

protected:
  virtual void emitDiagnosticMessage(SourceLocation Loc,PresumedLoc PLoc,
                                     DiagnosticsEngine::Level Level,
                                     StringRef Message,
                                     ArrayRef<CharSourceRange> Ranges,
                                     const SourceManager *SM,
                                     DiagOrStoredDiag D);

  virtual void emitDiagnosticLoc(SourceLocation Loc, PresumedLoc PLoc,
                                 DiagnosticsEngine::Level Level,
                                 ArrayRef<CharSourceRange> Ranges,
                                 const SourceManager &SM);
  
  virtual void emitCodeContext(SourceLocation Loc,
                               DiagnosticsEngine::Level Level,
                               SmallVectorImpl<CharSourceRange>& Ranges,
                               ArrayRef<FixItHint> Hints,
                               const SourceManager &SM) {
    emitSnippetAndCaret(Loc, Level, Ranges, Hints, SM);
  }
  
  virtual void emitBasicNote(StringRef Message);
  
  virtual void emitIncludeLocation(SourceLocation Loc, PresumedLoc PLoc,
                                   const SourceManager &SM);

private:
  void emitSnippetAndCaret(SourceLocation Loc, DiagnosticsEngine::Level Level,
                           SmallVectorImpl<CharSourceRange>& Ranges,
                           ArrayRef<FixItHint> Hints,
                           const SourceManager &SM);

  void emitSnippet(StringRef SourceLine);

  void highlightRange(const CharSourceRange &R,
                      unsigned LineNo, FileID FID,
                      const SourceColumnMap &map,
                      std::string &CaretLine,
                      const SourceManager &SM);

  std::string buildFixItInsertionLine(unsigned LineNo,
                                      const SourceColumnMap &map,
                                      ArrayRef<FixItHint> Hints,
                                      const SourceManager &SM);
  void emitParseableFixits(ArrayRef<FixItHint> Hints, const SourceManager &SM);
};

} // end namespace clang

#endif
