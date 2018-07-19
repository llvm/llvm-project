//==-- handle_cxx.cpp - Helper function for Clang fuzzers ------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements HandleCXX for use by the Clang fuzzers.
//
//===----------------------------------------------------------------------===//

#include "handle_cxx.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/TargetSelect.h"

using namespace clang;

void clang_fuzzer::HandleCXX(const std::string &S,
                             const std::vector<const char *> &ExtraArgs) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::opt::ArgStringList CC1Args;
  CC1Args.push_back("-cc1");
  for (auto &A : ExtraArgs)
    CC1Args.push_back(A);
  CC1Args.push_back("./test.cc");

  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions()));
  IgnoringDiagConsumer Diags;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<clang::DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &Diags, false);
  std::unique_ptr<clang::CompilerInvocation> Invocation(
      tooling::newInvocation(&Diagnostics, CC1Args));
  std::unique_ptr<llvm::MemoryBuffer> Input =
      llvm::MemoryBuffer::getMemBuffer(S);
  Invocation->getPreprocessorOpts().addRemappedFile("./test.cc",
                                                    Input.release());
  std::unique_ptr<tooling::ToolAction> action(
      tooling::newFrontendActionFactory<clang::EmitObjAction>());
  std::shared_ptr<PCHContainerOperations> PCHContainerOps =
      std::make_shared<PCHContainerOperations>();
  action->runInvocation(std::move(Invocation), Files.get(), PCHContainerOps,
                        &Diags);
}

