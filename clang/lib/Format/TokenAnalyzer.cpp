//===--- TokenAnalyzer.cpp - Analyze Token Streams --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements an abstract TokenAnalyzer and associated helper
/// classes. TokenAnalyzer can be extended to generate replacements based on
/// an annotated and pre-processed token stream.
///
//===----------------------------------------------------------------------===//

#include "TokenAnalyzer.h"
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

#define DEBUG_TYPE "format-formatter"

namespace clang {
namespace format {

// This sets up an virtual file system with file \p FileName containing \p
// Code.
std::unique_ptr<Environment>
Environment::CreateVirtualEnvironment(StringRef Code, StringRef FileName,
                                      ArrayRef<tooling::Range> Ranges) {
  // This is referenced by `FileMgr` and will be released by `FileMgr` when it
  // is deleted.
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
      new vfs::InMemoryFileSystem);
  // This is passed to `SM` as reference, so the pointer has to be referenced
  // in `Environment` so that `FileMgr` can out-live this function scope.
  std::unique_ptr<FileManager> FileMgr(
      new FileManager(FileSystemOptions(), InMemoryFileSystem));
  // This is passed to `SM` as reference, so the pointer has to be referenced
  // by `Environment` due to the same reason above.
  std::unique_ptr<DiagnosticsEngine> Diagnostics(new DiagnosticsEngine(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions));
  // This will be stored as reference, so the pointer has to be stored in
  // due to the same reason above.
  std::unique_ptr<SourceManager> VirtualSM(
      new SourceManager(*Diagnostics, *FileMgr));
  InMemoryFileSystem->addFile(
      FileName, 0, llvm::MemoryBuffer::getMemBuffer(
                       Code, FileName, /*RequiresNullTerminator=*/false));
  FileID ID = VirtualSM->createFileID(FileMgr->getFile(FileName),
                                      SourceLocation(), clang::SrcMgr::C_User);
  assert(ID.isValid());
  SourceLocation StartOfFile = VirtualSM->getLocForStartOfFile(ID);
  std::vector<CharSourceRange> CharRanges;
  for (const tooling::Range &Range : Ranges) {
    SourceLocation Start = StartOfFile.getLocWithOffset(Range.getOffset());
    SourceLocation End = Start.getLocWithOffset(Range.getLength());
    CharRanges.push_back(CharSourceRange::getCharRange(Start, End));
  }
  return llvm::make_unique<Environment>(ID, std::move(FileMgr),
                                        std::move(VirtualSM),
                                        std::move(Diagnostics), CharRanges);
}

TokenAnalyzer::TokenAnalyzer(const Environment &Env, const FormatStyle &Style)
    : Style(Style), Env(Env),
      AffectedRangeMgr(Env.getSourceManager(), Env.getCharRanges()),
      UnwrappedLines(1),
      Encoding(encoding::detectEncoding(
          Env.getSourceManager().getBufferData(Env.getFileID()))) {
  DEBUG(
      llvm::dbgs() << "File encoding: "
                   << (Encoding == encoding::Encoding_UTF8 ? "UTF8" : "unknown")
                   << "\n");
  DEBUG(llvm::dbgs() << "Language: " << getLanguageName(Style.Language)
                     << "\n");
}

tooling::Replacements TokenAnalyzer::process() {
  tooling::Replacements Result;
  FormatTokenLexer Tokens(Env.getSourceManager(), Env.getFileID(), Style,
                          Encoding);

  UnwrappedLineParser Parser(Style, Tokens.getKeywords(), Tokens.lex(), *this);
  Parser.parse();
  assert(UnwrappedLines.rbegin()->empty());
  for (unsigned Run = 0, RunE = UnwrappedLines.size(); Run + 1 != RunE; ++Run) {
    DEBUG(llvm::dbgs() << "Run " << Run << "...\n");
    SmallVector<AnnotatedLine *, 16> AnnotatedLines;

    TokenAnnotator Annotator(Style, Tokens.getKeywords());
    for (unsigned i = 0, e = UnwrappedLines[Run].size(); i != e; ++i) {
      AnnotatedLines.push_back(new AnnotatedLine(UnwrappedLines[Run][i]));
      Annotator.annotate(*AnnotatedLines.back());
    }

    tooling::Replacements RunResult =
        analyze(Annotator, AnnotatedLines, Tokens, Result);

    DEBUG({
      llvm::dbgs() << "Replacements for run " << Run << ":\n";
      for (tooling::Replacements::iterator I = RunResult.begin(),
                                           E = RunResult.end();
           I != E; ++I) {
        llvm::dbgs() << I->toString() << "\n";
      }
    });
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      delete AnnotatedLines[i];
    }
    Result.insert(RunResult.begin(), RunResult.end());
  }
  return Result;
}

void TokenAnalyzer::consumeUnwrappedLine(const UnwrappedLine &TheLine) {
  assert(!UnwrappedLines.empty());
  UnwrappedLines.back().push_back(TheLine);
}

void TokenAnalyzer::finishRun() {
  UnwrappedLines.push_back(SmallVector<UnwrappedLine, 16>());
}

} // end namespace format
} // end namespace clang
