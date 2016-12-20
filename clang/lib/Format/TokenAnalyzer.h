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
      : ID(ID), CharRanges(Ranges.begin(), Ranges.end()), SM(SM) {}

  Environment(FileID ID, std::unique_ptr<FileManager> FileMgr,
              std::unique_ptr<SourceManager> VirtualSM,
              std::unique_ptr<DiagnosticsEngine> Diagnostics,
              const std::vector<CharSourceRange> &CharRanges)
      : ID(ID), CharRanges(CharRanges.begin(), CharRanges.end()),
        SM(*VirtualSM), FileMgr(std::move(FileMgr)),
        VirtualSM(std::move(VirtualSM)), Diagnostics(std::move(Diagnostics)) {}

  // This sets up an virtual file system with file \p FileName containing \p
  // Code.
  static std::unique_ptr<Environment>
  CreateVirtualEnvironment(StringRef Code, StringRef FileName,
                           ArrayRef<tooling::Range> Ranges);

  FileID getFileID() const { return ID; }

  ArrayRef<CharSourceRange> getCharRanges() const { return CharRanges; }

  const SourceManager &getSourceManager() const { return SM; }

private:
  FileID ID;
  SmallVector<CharSourceRange, 8> CharRanges;
  SourceManager &SM;

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

  tooling::Replacements process();

protected:
  virtual tooling::Replacements
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
