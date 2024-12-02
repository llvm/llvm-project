//===---- ClangReadDiagnostics.cpp - clang-read-diagnostics tool -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool is for reading clang diagnostics files from -serialize-diagnostics.
//
// Example usage:
//
// $ clang -serialize-diagnostics foo.c.diag foo.c
// $ clang-read-diagnostics foo.c.diag
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/LineEditor/LineEditor.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "clang/Frontend/SerializedDiagnosticReader.h"

#include <optional>
#include <string>

using namespace clang;
using namespace clang::serialized_diags;
using namespace llvm;

static cl::list<std::string> InputFiles(cl::Sink, cl::desc("<input files...>"), cl::Required);

class BasicSerializedDiagnosticReader : public SerializedDiagnosticReader {

protected:
  virtual std::error_code
  visitDiagnosticRecord(unsigned Severity, const Location &Location,
                        unsigned Category, unsigned Flag, StringRef Message) override {
    llvm::dbgs() << Message << "\n";
    return {};
  }
};

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  BasicSerializedDiagnosticReader BSDR{};
  for (const auto &File : InputFiles)
    BSDR.readDiagnostics(File);

  return 0;
}
