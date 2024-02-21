//===-- ClangInstallAPI.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to clang-installapi; it is a wrapper
// for functionality in the InstallAPI clang library.
//
//===----------------------------------------------------------------------===//

#include "Options.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/InstallAPI/Context.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TextAPI/TextAPIWriter.h"

using namespace clang;
using namespace clang::installapi;
using namespace clang::driver::options;
using namespace llvm::opt;
using namespace llvm::MachO;

static bool run(ArrayRef<const char *> Args, const char *ProgName) {
  // Setup Diagnostics engine.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  const llvm::opt::OptTable &ClangOpts = clang::driver::getDriverOptTable();
  unsigned MissingArgIndex, MissingArgCount;
  llvm::opt::InputArgList ParsedArgs = ClangOpts.ParseArgs(
      ArrayRef(Args).slice(1), MissingArgIndex, MissingArgCount);
  ParseDiagnosticArgs(*DiagOpts, ParsedArgs);

  IntrusiveRefCntPtr<DiagnosticsEngine> Diag = new clang::DiagnosticsEngine(
      new clang::DiagnosticIDs(), DiagOpts.get(),
      new clang::TextDiagnosticPrinter(llvm::errs(), DiagOpts.get()));

  // Create file manager for all file operations.
  IntrusiveRefCntPtr<clang::FileManager> FM(
      new FileManager(clang::FileSystemOptions()));

  // Set up driver to parse input arguments.
  auto DriverArgs = llvm::ArrayRef(Args).slice(1);
  clang::driver::Driver Driver(ProgName, llvm::sys::getDefaultTargetTriple(),
                               *Diag, "clang installapi tool");
  Driver.setInstalledDir(llvm::sys::path::parent_path(ProgName));
  auto TargetAndMode =
      clang::driver::ToolChain::getTargetAndModeFromProgramName(ProgName);
  Driver.setTargetAndMode(TargetAndMode);
  bool HasError = false;
  llvm::opt::InputArgList ArgList =
      Driver.ParseArgStrings(DriverArgs, /*UseDriverMode=*/true, HasError);
  if (HasError)
    return EXIT_FAILURE;
  Driver.setCheckInputsExist(false);

  // Capture InstallAPI specific options and diagnose any option errors.
  Options Opts(*Diag, FM.get(), ArgList);
  if (Diag->hasErrorOccurred())
    return EXIT_FAILURE;
  InstallAPIContext Ctx = Opts.createContext();

  // Set up compilation.
  std::unique_ptr<CompilerInstance> CI(new CompilerInstance());
  CI->setFileManager(FM.get());
  CI->createDiagnostics();
  if (!CI->hasDiagnostics())
    return EXIT_FAILURE;

  auto Out = CI->createOutputFile(Ctx.OutputLoc, /*Binary=*/false,
                                  /*RemoveFileOnSignal=*/false,
                                  /*UseTemporary=*/false,
                                  /*CreateMissingDirectories=*/false);
  if (!Out)
    return EXIT_FAILURE;

  // Assign attributes for serialization.
  InterfaceFile IF;
  for (const auto &TargetInfo : Opts.DriverOpts.Targets) {
    IF.addTarget(TargetInfo.first);
    IF.setFromBinaryAttrs(Ctx.BA, TargetInfo.first);
  }

  // Write output file and perform CI cleanup.
  if (auto Err = TextAPIWriter::writeToStream(*Out, IF, Ctx.FT)) {
    Diag->Report(diag::err_cannot_open_file) << Ctx.OutputLoc;
    CI->clearOutputFiles(/*EraseFiles=*/true);
    return EXIT_FAILURE;
  }

  CI->clearOutputFiles(/*EraseFiles=*/false);
  return EXIT_SUCCESS;
}

int clang_installapi_main(int argc, char **argv,
                          const llvm::ToolContext &ToolContext) {
  // Standard set up, so program fails gracefully.
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram StackPrinter(argc, argv);
  llvm::llvm_shutdown_obj Shutdown;

  if (llvm::sys::Process::FixupStandardFileDescriptors())
    return EXIT_FAILURE;

  const char *ProgName =
      ToolContext.NeedsPrependArg ? ToolContext.PrependArg : ToolContext.Path;
  return run(llvm::ArrayRef(argv, argc), ProgName);
}
