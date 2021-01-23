//===-- cc1as_main.cpp - Clang Assembler  ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1as functionality, which implements
// the direct interface to the LLVM MC based assembler.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/AssemblerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>
using namespace clang;
using namespace llvm;
using namespace llvm::opt;

static void LLVMErrorHandler(void *UserData, const std::string &Message,
                             bool GenCrashDiag) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  sys::Process::Exit(1);
}

int cc1as_main(ArrayRef<const char *> Argv, const char *Argv0, void *MainAddr) {
  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();

  // Construct our diagnostic client.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient
    = new TextDiagnosticPrinter(errs(), &*DiagOpts);
  DiagClient->setPrefix("clang -cc1as");
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  ScopedFatalErrorHandler FatalErrorHandler
    (LLVMErrorHandler, static_cast<void*>(&Diags));

  // Parse the arguments.
  AssemblerInvocation Asm;
  if (!AssemblerInvocation::CreateFromArgs(Asm, Argv, Diags))
    return 1;

  if (Asm.ShowHelp) {
    clang::driver::getDriverOptTable().PrintHelp(
        llvm::outs(), "clang -cc1as [options] file...",
        "Clang Integrated Assembler",
        /*Include=*/driver::options::CC1AsOption, /*Exclude=*/0,
        /*ShowAllAliases=*/false);
    return 0;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Asm.ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  if (!Asm.LLVMArgs.empty()) {
    unsigned NumArgs = Asm.LLVMArgs.size();
    auto Args = std::make_unique<const char*[]>(NumArgs + 2);
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Asm.LLVMArgs[i].c_str();
    Args[NumArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args.get());
  }

  // Execute the invocation, unless there were parsing errors.
  bool Failed = Diags.hasErrorOccurred() || ExecuteAssembler(Asm, Diags);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.
  TimerGroup::printAll(errs());
  TimerGroup::clearAll();

  return !!Failed;
}
