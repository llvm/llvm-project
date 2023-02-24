//===- ClangCASTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"

using namespace clang;
using namespace clang::cas;
using namespace llvm;

static ExitOnError ExitOnErr("clang-cas-test: ");

static void printCompileJobCacheKey(llvm::cas::ObjectStore &CAS,
                                    StringRef Key) {
  auto KeyID = ExitOnErr(CAS.parseID(Key));
  ExitOnErr(printCompileJobCacheKey(CAS, KeyID, outs()));
}

static void printIncludeTree(llvm::cas::ObjectStore &CAS, StringRef Key) {
  auto ID = ExitOnErr(CAS.parseID(Key));
  llvm::cas::ObjectRef Ref = *CAS.getReference(ID);
  auto IncludeTree = ExitOnErr(IncludeTreeRoot::get(CAS, Ref));
  ExitOnErr(IncludeTree.print(outs()));
}

int main(int Argc, const char **Argv) {
  llvm::InitLLVM X(Argc, Argv);

  enum ActionType {
    None,
    PrintCompileJobCacheKey,
    PrintIncludeTree,
  };

  cl::opt<ActionType> Action(
      cl::desc("Action:"), cl::init(ActionType::None),
      cl::values(clEnumValN(PrintCompileJobCacheKey,
                            "print-compile-job-cache-key",
                            "Print a compile-job result cache key's structure"),
                 clEnumValN(PrintIncludeTree, "print-include-tree",
                            "Print include tree structure")));

  cl::opt<std::string> CASPath("cas", cl::desc("On-disk CAS path"),
                               cl::value_desc("path"));
  cl::list<std::string> Inputs(cl::Positional, cl::desc("<input>..."));

  cl::ParseCommandLineOptions(Argc, Argv, "clang-cas-test");

  if (Action == ActionType::None) {
    errs() << "error: action required; pass '-help' for options\n";
    return 1;
  }

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
      CompilerInstance::createDiagnostics(new DiagnosticOptions));

  CASOptions Opts;
  // Printing a cache key only makes sense in an existing CAS, so default to
  // on-disk instead of in-memory if no --cas path is specified.
  Opts.CASPath = CASPath.empty() ? std::string("auto") : CASPath;

  auto CAS =
      Opts.getOrCreateDatabases(*Diags, /*CreateEmptyCASOnFailure=*/false)
          .first;
  if (!CAS)
    return 1;

  if (Action == ActionType::PrintCompileJobCacheKey) {
    if (Inputs.empty()) {
      errs() << "error: missing compile-job cache key in inputs'\n";
      return 1;
    }
    for (StringRef CacheKey : Inputs)
      printCompileJobCacheKey(*CAS, CacheKey);
  } else if (Action == ActionType::PrintIncludeTree) {
    if (Inputs.empty()) {
      errs() << "error: missing include tree ID in inputs\n";
      return 1;
    }
    for (StringRef CacheKey : Inputs)
      printIncludeTree(*CAS, CacheKey);
  }

  return 0;
}
