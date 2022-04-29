//===--- ExecuteCompilerInvocation.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds ExecuteCompilerInvocation(). It is split into its own file to
// minimize the impact of pulling in essentially everything else in Flang.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CommandLine.h"

namespace Fortran::frontend {

static std::unique_ptr<FrontendAction>
createFrontendAction(CompilerInstance &ci) {

  switch (ci.getFrontendOpts().programAction) {
  case InputOutputTest:
    return std::make_unique<InputOutputTestAction>();
  case PrintPreprocessedInput:
    return std::make_unique<PrintPreprocessedAction>();
  case ParseSyntaxOnly:
    return std::make_unique<ParseSyntaxOnlyAction>();
  case EmitMLIR:
    return std::make_unique<EmitMLIRAction>();
  case EmitLLVM:
    return std::make_unique<EmitLLVMAction>();
  case EmitLLVMBitcode:
    return std::make_unique<EmitLLVMBitcodeAction>();
  case EmitObj:
    return std::make_unique<EmitObjAction>();
  case EmitAssembly:
    return std::make_unique<EmitAssemblyAction>();
  case DebugUnparse:
    return std::make_unique<DebugUnparseAction>();
  case DebugUnparseNoSema:
    return std::make_unique<DebugUnparseNoSemaAction>();
  case DebugUnparseWithSymbols:
    return std::make_unique<DebugUnparseWithSymbolsAction>();
  case DebugDumpSymbols:
    return std::make_unique<DebugDumpSymbolsAction>();
  case DebugDumpParseTree:
    return std::make_unique<DebugDumpParseTreeAction>();
  case DebugDumpPFT:
    return std::make_unique<DebugDumpPFTAction>();
  case DebugDumpParseTreeNoSema:
    return std::make_unique<DebugDumpParseTreeNoSemaAction>();
  case DebugDumpAll:
    return std::make_unique<DebugDumpAllAction>();
  case DebugDumpProvenance:
    return std::make_unique<DebugDumpProvenanceAction>();
  case DebugDumpParsingLog:
    return std::make_unique<DebugDumpParsingLogAction>();
  case DebugMeasureParseTree:
    return std::make_unique<DebugMeasureParseTreeAction>();
  case DebugPreFIRTree:
    return std::make_unique<DebugPreFIRTreeAction>();
  case GetDefinition:
    return std::make_unique<GetDefinitionAction>();
  case GetSymbolsSources:
    return std::make_unique<GetSymbolsSourcesAction>();
  case InitOnly:
    return std::make_unique<InitOnlyAction>();
  case PluginAction: {
    for (const FrontendPluginRegistry::entry &plugin :
        FrontendPluginRegistry::entries()) {
      if (plugin.getName() == ci.getFrontendOpts().actionName) {
        std::unique_ptr<PluginParseTreeAction> p(plugin.instantiate());
        return std::move(p);
      }
    }
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "unable to find plugin '%0'");
    ci.getDiagnostics().Report(diagID) << ci.getFrontendOpts().actionName;
    return nullptr;
  }
  }

  llvm_unreachable("Invalid program action!");
}

bool executeCompilerInvocation(CompilerInstance *flang) {
  // Honor -help.
  if (flang->getFrontendOpts().showHelp) {
    clang::driver::getDriverOptTable().printHelp(llvm::outs(),
        "flang-new -fc1 [options] file...", "LLVM 'Flang' Compiler",
        /*Include=*/clang::driver::options::FC1Option,
        /*Exclude=*/llvm::opt::DriverFlag::HelpHidden,
        /*ShowAllAliases=*/false);
    return true;
  }

  // Honor -version.
  if (flang->getFrontendOpts().showVersion) {
    llvm::cl::PrintVersionMessage();
    return true;
  }

  // Load any requested plugins.
  for (const std::string &path : flang->getFrontendOpts().plugins) {
    std::string error;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(),
                                                          &error)) {
      unsigned diagID = flang->getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "unable to load plugin '%0': '%1'");
      flang->getDiagnostics().Report(diagID) << path << error;
    }
  }

  // Honor -mllvm. This should happen AFTER plugins have been loaded!
  if (!flang->getFrontendOpts().llvmArgs.empty()) {
    unsigned numArgs = flang->getFrontendOpts().llvmArgs.size();
    auto args = std::make_unique<const char *[]>(numArgs + 2);
    args[0] = "flang (LLVM option parsing)";

    for (unsigned i = 0; i != numArgs; ++i)
      args[i + 1] = flang->getFrontendOpts().llvmArgs[i].c_str();

    args[numArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(numArgs + 1, args.get());
  }

  // Honor -mmlir. This should happen AFTER plugins have been loaded!
  if (!flang->getFrontendOpts().mlirArgs.empty()) {
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    unsigned numArgs = flang->getFrontendOpts().mlirArgs.size();
    auto args = std::make_unique<const char *[]>(numArgs + 2);
    args[0] = "flang (MLIR option parsing)";

    for (unsigned i = 0; i != numArgs; ++i)
      args[i + 1] = flang->getFrontendOpts().mlirArgs[i].c_str();

    args[numArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(numArgs + 1, args.get());
  }

  // If there were errors in processing arguments, don't do anything else.
  if (flang->getDiagnostics().hasErrorOccurred()) {
    return false;
  }

  // Create and execute the frontend action.
  std::unique_ptr<FrontendAction> act(createFrontendAction(*flang));
  if (!act)
    return false;

  bool success = flang->executeAction(*act);
  return success;
}

} // namespace Fortran::frontend
