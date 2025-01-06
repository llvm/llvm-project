//===- bolt/tools/binary-analysis/binary-analysis.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a generic binary analysis tool, where multiple different specific
// binary analyses can be plugged in to. The binary analyses are mostly built
// on top of BOLT components.
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"

#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {

static cl::OptionCategory *BinaryAnalysisCategories[] = {
    &BinaryAnalysisCategory};

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<executable>"),
                                          cl::Required,
                                          cl::cat(BinaryAnalysisCategory),
                                          cl::sub(cl::SubCommand::getAll()));

} // namespace opts

static StringRef ToolName = "llvm-bolt-binary-analysis";

static void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

static void report_error(StringRef Message, Error E) {
  assert(E);
  errs() << ToolName << ": '" << Message << "': " << toString(std::move(E))
         << ".\n";
  exit(1);
}

void ParseCommandLine(int argc, char **argv) {
  cl::HideUnrelatedOptions(ArrayRef(opts::BinaryAnalysisCategories));
  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "BinaryAnalysis\n");
}

static std::string GetExecutablePath(const char *Argv0) {
  SmallString<256> ExecutablePath(Argv0);
  // Do a PATH lookup if Argv0 isn't a valid path.
  if (!llvm::sys::fs::exists(ExecutablePath))
    if (llvm::ErrorOr<std::string> P =
            llvm::sys::findProgramByName(ExecutablePath))
      ExecutablePath = *P;
  return std::string(ExecutablePath.str());
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  std::string ToolPath = GetExecutablePath(argv[0]);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  ParseCommandLine(argc, argv);

  opts::BinaryAnalysisMode = true;

  if (!sys::fs::exists(opts::InputFilename))
    report_error(opts::InputFilename, errc::no_such_file_or_directory);

  Expected<OwningBinary<Binary>> BinaryOrErr =
      createBinary(opts::InputFilename);
  if (Error E = BinaryOrErr.takeError())
    report_error(opts::InputFilename, std::move(E));
  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (auto *e = dyn_cast<ELFObjectFileBase>(&Binary)) {
    auto RIOrErr = RewriteInstance::create(e, argc, argv, ToolPath);
    if (Error E = RIOrErr.takeError())
      report_error(opts::InputFilename, std::move(E));
    RewriteInstance &RI = *RIOrErr.get();
    if (Error E = RI.run())
      report_error(opts::InputFilename, std::move(E));
  }

  return EXIT_SUCCESS;
}
