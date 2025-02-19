//===-- fc1_main.cpp - Flang FC1 Compiler Frontend ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the flang -fc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticBuffer.h"
#include "flang/FrontendTool/Utils.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/X86TargetParser.h"

#include <cstdio>

using namespace Fortran::frontend;

/// Instantiate llvm::Target based on triple
static const llvm::Target* getTarget(llvm::StringRef triple) {
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << error;
  }

  return target;
}

/// Print supported cpus of the given target.
static int printSupportedCPUs(llvm::StringRef triple) {
  const llvm::Target *target = getTarget(triple);
  if (!target) {
    return 1;
  }

  // the target machine will handle the mcpu printing
  llvm::TargetOptions targetOpts;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(triple, "", "+cpuhelp", targetOpts,
                                  std::nullopt));
  return 0;
}

/// Check that given CPU is valid for given target.
static bool checkSupportedCPU(llvm::StringRef str_cpu, llvm::StringRef str_triple) {
  // If can create tareg from triple, then it's a valid triple
  const llvm::Target *target = getTarget(str_triple);
  if (!target) {
    return false;
  }

  // TODO: only support check for x86_64 for now
  llvm::Triple triple{str_triple};
  if (triple.getArch() == llvm::Triple::x86_64) {
    const bool only64bit{true};
    llvm::X86::CPUKind x86cpu = llvm::X86::parseArchX86(str_cpu, only64bit);
    return x86cpu != llvm::X86::CK_None;
  }
  else {
    // TODO: only support check for x86_64 for now. Anything else passes.
    return true;
  }
}

int fc1_main(llvm::ArrayRef<const char *> argv, const char *argv0) {
  // Create CompilerInstance
  std::unique_ptr<CompilerInstance> flang(new CompilerInstance());

  // Create DiagnosticsEngine for the frontend driver
  flang->createDiagnostics();
  if (!flang->hasDiagnostics())
    return 1;

  // We will buffer diagnostics from argument parsing so that we can output
  // them using a well formed diagnostic object.
  TextDiagnosticBuffer *diagsBuffer = new TextDiagnosticBuffer;

  // Create CompilerInvocation - use a dedicated instance of DiagnosticsEngine
  // for parsing the arguments
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
      new clang::DiagnosticOptions();
  clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagsBuffer);
  bool success = CompilerInvocation::createFromArgs(flang->getInvocation(),
                                                    argv, diags, argv0);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  // --print-supported-cpus takes priority over the actual compilation.
  if (flang->getFrontendOpts().printSupportedCPUs)
    return printSupportedCPUs(flang->getInvocation().getTargetOpts().triple);

  // Check that requested CPU can be properly supported
  if (!checkSupportedCPU(flang->getInvocation().getTargetOpts().cpu, flang->getInvocation().getTargetOpts().triple))
    return 1;

  diagsBuffer->flushDiagnostics(flang->getDiagnostics());

  if (!success)
    return 1;

  // Execute the frontend actions.
  success = executeCompilerInvocation(flang.get());

  // Delete output files to free Compiler Instance
  flang->clearOutputFiles(/*EraseFiles=*/false);

  return !success;
}
