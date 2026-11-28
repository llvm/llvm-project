//===-- APINotesTest.cpp - API Notes Testing Tool ------------------ C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/APINotes/APINotesYAMLCompiler.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

namespace {
using namespace llvm::opt;
enum ID {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_CODE
#include "Opts.inc"

class APINotesTestOptTable : public OptTable {
public:
  APINotesTestOptTable() : OptTable(optionTables()) {
    setDashDashParsing(true);
  }
};
} // namespace

int main(int argc, const char **argv) {
  const bool DisableCrashReporting = true;
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0], DisableCrashReporting);

  auto Error = [](const llvm::Twine &Msg) {
    llvm::WithColor::error(llvm::errs(), "apinotes-test") << Msg << '\n';
  };

  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  APINotesTestOptTable Tbl;
  llvm::opt::InputArgList Args =
      Tbl.parseArgs(argc, const_cast<char **>(argv), OPT_UNKNOWN, Saver,
                    [&](llvm::StringRef Msg) {
                      Error(Msg);
                      exit(EXIT_FAILURE);
                    });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(llvm::outs(), "apinotes-test [options] <apinotes> ...",
                  "API notes testing tool");
    return EXIT_SUCCESS;
  }
  std::vector<std::string> APINotes = Args.getAllArgValues(OPT_INPUT);
  if (APINotes.empty()) {
    Error("no input files");
    return EXIT_FAILURE;
  }
  std::string OutputFileName = Args.getLastArgValue(OPT_o, "-").str();

  std::error_code EC;
  auto Out = std::make_unique<llvm::ToolOutputFile>(OutputFileName, EC,
                                                    llvm::sys::fs::OF_None);
  if (EC) {
    Error("failed to open '" + OutputFileName + "': " + EC.message());
    return EXIT_FAILURE;
  }

  for (const std::string &Notes : APINotes) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> NotesOrError =
        llvm::MemoryBuffer::getFileOrSTDIN(Notes);
    if (std::error_code EC = NotesOrError.getError()) {
      llvm::errs() << EC.message() << '\n';
      return EXIT_FAILURE;
    }

    clang::api_notes::parseAndDumpAPINotes((*NotesOrError)->getBuffer(),
                                           Out->os());
  }

  return EXIT_SUCCESS;
}
