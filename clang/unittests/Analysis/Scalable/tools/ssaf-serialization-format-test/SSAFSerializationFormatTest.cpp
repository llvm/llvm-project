//===- SSAFSerializationFormatTest.cpp - Test SSAF JSON serialization ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool reads SSAF TUSummary JSON files and can:
// - Validate the JSON format
// - Print summary information
// - Write back to JSON (for round-trip testing)
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/WithColor.h"

using namespace clang::ssaf;
using namespace llvm;

static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> PrintSummary("print-summary",
                                  cl::desc("Print summary information"),
                                  cl::init(false));

static cl::opt<bool> Quiet("q", cl::desc("Suppress diagnostic messages"),
                           cl::init(false));

static void reportError(StringRef Prefix, Error E) {
  if (Quiet)
    return;
  WithColor::error(errs(), "ssaf-serialization-format-test")
      << Prefix << ": " << toString(std::move(E)) << "\n";
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "SSAF JSON format validator and dumper\n");

  // Create JSONFormat instance
  auto FS = vfs::getRealFileSystem();
  JSONFormat Format(FS);

  // Read the input file
  auto SummaryOrErr = Format.readTUSummary(InputFilename);
  if (!SummaryOrErr) {
    reportError("failed to read TUSummary", SummaryOrErr.takeError());
    return 1;
  }

  TUSummary &Summary = *SummaryOrErr;

  // Print summary information if requested
  if (PrintSummary) {
    outs() << "TUSummary successfully read from: " << InputFilename << "\n";
    // Could add more detailed summary info here
  }

  // Write output if specified
  if (OutputFilename != "-") {
    if (auto Err = Format.writeTUSummary(Summary, OutputFilename)) {
      reportError("failed to write TUSummary", std::move(Err));
      return 1;
    }
    if (!Quiet) {
      outs() << "TUSummary written to: " << OutputFilename << "\n";
    }
  }

  return 0;
}
