//===-- cc1_main.cpp - Clang CC1 Compiler Frontend ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/LinkAllPasses.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

int cc1_main(const char **ArgBegin, const char **ArgEnd,
             const char *Argv0, void *MainAddr) {
  OwningPtr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);
  bool Success;
  Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                               ArgBegin, ArgEnd, Diags);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics(ArgEnd - ArgBegin, const_cast<char**>(ArgBegin));
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler,
                                  static_cast<void*>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success)
    return 1;

  // Execute the frontend actions.
  Success = ExecuteCompilerInvocation(Clang.get());

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (Clang->getFrontendOpts().DisableFree) {
    if (llvm::AreStatisticsEnabled() || Clang->getFrontendOpts().ShowStats)
      llvm::PrintStatistics();
    Clang.take();
    return !Success;
  }

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return !Success;
}
