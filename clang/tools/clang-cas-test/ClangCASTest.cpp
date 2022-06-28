//===- ClangCASTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"

using namespace clang;
using namespace llvm;

static void printCompileJobCacheKey(StringRef CASPath, StringRef Key) {
  ExitOnError ExitOnErr;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
      CompilerInstance::createDiagnostics(new DiagnosticOptions));

  CASOptions Opts;
  // Printing a cache key only makes sense in an existing CAS, so default to
  // on-disk instead of in-memory if no --cas path is specified.
  Opts.CASPath = CASPath.empty() ? "auto" : CASPath.str();

  auto CAS = Opts.getOrCreateCAS(*Diags, /*CreateEmptyCASOnFailure=*/false);
  if (!CAS)
    return;

  auto KeyID = ExitOnErr(CAS->parseID(Key));
  ExitOnErr(printCompileJobCacheKey(*CAS, KeyID, outs()));
}

int main(int Argc, const char **Argv) {
  llvm::InitLLVM X(Argc, Argv);

  enum ActionType {
    None,
    PrintCompileJobCacheKey,
  };

  cl::opt<ActionType> Action(
      cl::desc("Action:"), cl::init(ActionType::None),
      cl::values(
          clEnumValN(PrintCompileJobCacheKey, "print-compile-job-cache-key",
                     "Print a compile-job result cache key's structure")));

  cl::opt<std::string> CASPath("cas", cl::desc("On-disk CAS path"),
                               cl::value_desc("path"));
  cl::list<std::string> Inputs(cl::Positional, cl::desc("<input>..."));

  cl::ParseCommandLineOptions(Argc, Argv, "clang-cas-test");
  ExitOnError ExitOnErr("clang-cas-test: ");

  if (Action == ActionType::None) {
    errs() << "error: action required; pass '-help' for options\n";
    return 1;
  }

  if (Action == ActionType::PrintCompileJobCacheKey) {
    if (Inputs.empty()) {
      errs() << "error: missing compile-job cache key in inputs'\n";
      return 1;
    }
    for (StringRef CacheKey : Inputs)
      printCompileJobCacheKey(CASPath, CacheKey);
  }

  return 0;
}
