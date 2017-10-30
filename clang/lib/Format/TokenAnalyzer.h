//===--- TokenAnalyzer.h - Analyze Token Streams ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares an abstract TokenAnalyzer, and associated helper
/// classes. TokenAnalyzer can be extended to generate replacements based on
/// an annotated and pre-processed token stream.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_TOKENANALYZER_H
#define LLVM_CLANG_LIB_FORMAT_TOKENANALYZER_H

#include "AffectedRangeManager.h"
#include "Encoding.h"
#include "FormatToken.h"
#include "FormatTokenLexer.h"
#include "TokenAnnotator.h"
#include "UnwrappedLineParser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

namespace clang {
namespace format {

class Environment {
public:
  Environment(SourceManager &SM, FileID ID, ArrayRef<CharSourceRange> Ranges)
      : ID(ID), CharRanges(Ranges.begin(), Ranges.end()), SM(SM),
      FirstStartColumn(0),
      NextStartColumn(0),
      LastStartColumn(0) {}

  Environment(FileID ID, std::unique_ptr<FileManager> FileMgr,
              std::unique_ptr<SourceManager> VirtualSM,
              std::unique_ptr<DiagnosticsEngine> Diagnostics,
              const std::vector<CharSourceRange> &CharRanges,
              unsigned FirstStartColumn,
              unsigned NextStartColumn,
              unsigned LastStartColumn)
      : ID(ID), CharRanges(CharRanges.begin(), CharRanges.end()),
        SM(*VirtualSM), 
        FirstStartColumn(FirstStartColumn),
        NextStartColumn(NextStartColumn),
        LastStartColumn(LastStartColumn),
        FileMgr(std::move(FileMgr)),
        VirtualSM(std::move(VirtualSM)), Diagnostics(std::move(Diagnostics)) {}

  // This sets up an virtual file system with file \p FileName containing the
  // fragment \p Code. Assumes that \p Code starts at \p FirstStartColumn,
  // that the next lines of \p Code should start at \p NextStartColumn, and
  // that \p Code should end at \p LastStartColumn if it ends in newline.
  // See also the documentation of clang::format::internal::reformat.
  static std::unique_ptr<Environment>
  CreateVirtualEnvironment(StringRef Code, StringRef FileName,
                           ArrayRef<tooling::Range> Ranges,
                           unsigned FirstStartColumn = 0,
                           unsigned NextStartColumn = 0,
                           unsigned LastStartColumn = 0);

  FileID getFileID() const { return ID; }

  ArrayRef<CharSourceRange> getCharRanges() const { return CharRanges; }

  const SourceManager &getSourceManager() const { return SM; }

  // Returns the column at which the fragment of code managed by this
  // environment starts.
  unsigned getFirstStartColumn() const { return FirstStartColumn; }

  // Returns the column at which subsequent lines of the fragment of code
  // managed by this environment should start.
  unsigned getNextStartColumn() const { return NextStartColumn; }

  // Returns the column at which the fragment of code managed by this
  // environment should end if it ends in a newline.
  unsigned getLastStartColumn() const { return LastStartColumn; }

private:
  FileID ID;
  SmallVector<CharSourceRange, 8> CharRanges;
  SourceManager &SM;
  unsigned FirstStartColumn;
  unsigned NextStartColumn;
  unsigned LastStartColumn;

  // The order of these fields are important - they should be in the same order
  // as they are created in `CreateVirtualEnvironment` so that they can be
  // deleted in the reverse order as they are created.
  std::unique_ptr<FileManager> FileMgr;
  std::unique_ptr<SourceManager> VirtualSM;
  std::unique_ptr<DiagnosticsEngine> Diagnostics;
};

class TokenAnalyzer : public UnwrappedLineConsumer {
public:
  TokenAnalyzer(const Environment &Env, const FormatStyle &Style);

  std::pair<tooling::Replacements, unsigned> process();

protected:
  virtual std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) = 0;

  void consumeUnwrappedLine(const UnwrappedLine &TheLine) override;

  void finishRun() override;

  FormatStyle Style;
  // Stores Style, FileID and SourceManager etc.
  const Environment &Env;
  // AffectedRangeMgr stores ranges to be fixed.
  AffectedRangeManager AffectedRangeMgr;
  SmallVector<SmallVector<UnwrappedLine, 16>, 2> UnwrappedLines;
  encoding::Encoding Encoding;
};

} // end namespace format
} // end namespace clang

#endif
