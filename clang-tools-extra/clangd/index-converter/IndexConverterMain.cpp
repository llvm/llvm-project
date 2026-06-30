//===--- IndexConverterMain.cpp - Convert YAML index to RIFF ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// clangd-index-converter is a standalone tool that reads a YAML-format
// clangd index file and writes it out as a RIFF-format (binary) index file.
//
// Usage:
//   clangd-index-converter input.yaml --output=output.idx
//   clangd-index-converter input.yaml            # writes input.idx
//   clangd-index-converter input.yaml --output=- # writes to stdout
//
//===----------------------------------------------------------------------===//

#include "index/Serialization.h"
#include "support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace clang {
namespace clangd {
namespace {

static llvm::cl::OptionCategory ConverterCategory("clangd-index-converter options");

static llvm::cl::opt<std::string> Input{
    llvm::cl::desc("<input.yaml>"),
    llvm::cl::cat(ConverterCategory),
    llvm::cl::Positional,
    llvm::cl::Required,
};

static llvm::cl::opt<std::string> Output{
    "output",
    llvm::cl::cat(ConverterCategory),
    llvm::cl::desc("Output file path. Defaults to the input path with the "
                   "extension replaced by .idx. Use '-' for stdout."),
    llvm::cl::init(""),
};

static llvm::cl::opt<Logger::Level> LogLevel{
    "log",
    llvm::cl::cat(ConverterCategory),
    llvm::cl::desc("Verbosity of log messages written to stderr"),
    llvm::cl::values(
        clEnumValN(Logger::Error, "error", "Error messages only"),
        clEnumValN(Logger::Info, "info", "High level execution tracing"),
        clEnumValN(Logger::Debug, "verbose", "Low level details")),
    llvm::cl::init(Logger::Info),
};

// Derive output path from input path: replace extension with .idx.
std::string deriveOutputPath(llvm::StringRef InputPath) {
  llvm::SmallString<256> Out(InputPath);
  llvm::sys::path::replace_extension(Out, ".idx");
  return Out.str().str();
}

// Scan the YAML data and remove any document (--- ... ---) that contains a
// malformed record. Each document is buffered; if it contains the known bad
// pattern emitted by clangd-indexer for SymbolKind::Unknown ("Kind:  Lang:")
// it is dropped and counted. The cleaned YAML is returned as a new string.
std::string filterMalformedYAMLDocs(llvm::StringRef Data,
                                    unsigned &SkippedOut) {
  // Known bad pattern: clangd-indexer serializes SymbolKind::Unknown as an
  // empty scalar, causing the YAML formatter to merge "Kind:" and "Lang:" onto
  // one line, e.g. "  Kind:            Lang:            C".
  constexpr llvm::StringLiteral BadPattern("Kind:            Lang:");

  std::string Result;
  Result.reserve(Data.size());
  SkippedOut = 0;

  // Walk line by line, collecting each --- document into a temporary buffer
  // and flushing it only if it is clean.
  llvm::StringRef Remaining = Data;
  std::string DocBuf;
  DocBuf.reserve(4096);
  bool DocIsBad = false;

  while (!Remaining.empty()) {
    // Extract the next line (including the newline character).
    auto Pos = Remaining.find('\n');
    llvm::StringRef Line = (Pos == llvm::StringRef::npos)
                               ? Remaining
                               : Remaining.substr(0, Pos + 1);
    Remaining = Remaining.drop_front(Line.size());

    bool IsDocSeparator = Line.starts_with("---");
    if (IsDocSeparator && !DocBuf.empty()) {
      // Flush the previous document.
      if (DocIsBad)
        ++SkippedOut;
      else
        Result += DocBuf;
      DocBuf.clear();
      DocIsBad = false;
    }

    DocBuf += Line;
    if (Line.contains(BadPattern))
      DocIsBad = true;
  }

  // Flush the last document.
  if (!DocBuf.empty()) {
    if (DocIsBad)
      ++SkippedOut;
    else
      Result += DocBuf;
  }

  return Result;
}

} // namespace
} // namespace clangd
} // namespace clang

int main(int Argc, const char **Argv) {
  using namespace clang::clangd;

  llvm::sys::PrintStackTraceOnErrorSignal(Argv[0]);
  llvm::cl::HideUnrelatedOptions(ConverterCategory);
  llvm::cl::ParseCommandLineOptions(
      Argc, Argv,
      "clangd-index-converter: convert a YAML clangd index to RIFF format\n");

  StreamLogger Logger(llvm::errs(), LogLevel);
  LoggingSession LoggingSession(Logger);

  // Read input file.
  auto Buf = llvm::MemoryBuffer::getFile(Input);
  if (!Buf) {
    elog("Cannot read {0}: {1}", Input, Buf.getError().message());
    return 1;
  }

  llvm::StringRef Data = (*Buf)->getBuffer();

  // Warn if the input already looks like a RIFF binary index.
  if (Data.starts_with("RIFF")) {
    elog("{0} appears to already be a RIFF binary index; no conversion needed.",
         Input);
    return 1;
  }

  // Pre-process: remove documents containing malformed records (e.g. the
  // "Kind:  Lang:" pattern produced by clangd-indexer for SymbolKind::Unknown)
  // that would cause the YAML parser to abort.
  unsigned Skipped = 0;
  std::string Cleaned = filterMalformedYAMLDocs(Data, Skipped);
  if (Skipped)
    vlog("Skipped {0} malformed YAML document(s).", Skipped);
  llvm::StringRef ParseData = Skipped ? llvm::StringRef(Cleaned) : Data;

  // Parse the YAML index.
  auto Parsed = readIndexFile(ParseData, SymbolOrigin::Static);
  if (!Parsed) {
    elog("Failed to parse {0}: {1}", Input, Parsed.takeError());
    return 1;
  }

  if (Skipped)
    log("Read index from {0}: {1} symbol(s), {2} ref(s), {3} relation(s), "
        "{4} malformed record(s) skipped.",
        Input,
        Parsed->Symbols ? Parsed->Symbols->size() : 0u,
        Parsed->Refs ? Parsed->Refs->size() : 0u,
        Parsed->Relations ? Parsed->Relations->size() : 0u, Skipped);
  else
    log("Read index from {0}: {1} symbol(s), {2} ref(s), {3} relation(s).",
        Input,
        Parsed->Symbols ? Parsed->Symbols->size() : 0u,
        Parsed->Refs ? Parsed->Refs->size() : 0u,
        Parsed->Relations ? Parsed->Relations->size() : 0u);

  // Determine output path.
  std::string OutputPath =
      Output.empty() ? deriveOutputPath(Input) : Output.getValue();

  // Build the IndexFileOut pointing at the parsed slabs.
  IndexFileOut Out(*Parsed);
  Out.Format = IndexFileFormat::RIFF;

  // Write to stdout or a file.
  if (OutputPath == "-") {
    llvm::outs() << Out;
    log("Written RIFF index to stdout.");
    return 0;
  }

  if (auto Err = llvm::writeToOutput(OutputPath, [&](llvm::raw_ostream &OS) {
        OS << Out;
        return llvm::Error::success();
      })) {
    elog("Cannot write {0}: {1}", OutputPath, std::move(Err));
    return 1;
  }

  log("Written RIFF index to {0}.", OutputPath);
  return 0;
}
