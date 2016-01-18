//===--- CreateInvocationFromCommandLine.cpp - CompilerInvocation from Args ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Construct a compiler invocation object for command line driver arguments
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Host.h"
using namespace clang;
using namespace llvm::opt;

/// createInvocationFromCommandLine - Construct a compiler invocation object for
/// a command line argument vector.
///
/// \return A CompilerInvocation, or 0 if none was built for the given
/// argument vector.
CompilerInvocation *
clang::createInvocationFromCommandLine(ArrayRef<const char *> ArgList,
                            IntrusiveRefCntPtr<DiagnosticsEngine> Diags) {
  if (!Diags.get()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions);
  }

  SmallVector<const char *, 16> Args;
  Args.push_back("<clang>"); // FIXME: Remove dummy argument.
  Args.insert(Args.end(), ArgList.begin(), ArgList.end());

  // FIXME: Find a cleaner way to force the driver into restricted modes.
  Args.push_back("-fsyntax-only");

  // FIXME: We shouldn't have to pass in the path info.
  driver::Driver TheDriver("clang", llvm::sys::getDefaultTargetTriple(),
                           *Diags);

  // Don't check that inputs exist, they may have been remapped.
  TheDriver.setCheckInputsExist(false);

  std::unique_ptr<driver::Compilation> C(TheDriver.BuildCompilation(Args));

  // Just print the cc1 options if -### was present.
  if (C->getArgs().hasArg(driver::options::OPT__HASH_HASH_HASH)) {
    C->getJobs().Print(llvm::errs(), "\n", true);
    return nullptr;
  }

  // We expect to get back exactly one command job, if we didn't something
  // failed. CUDA compilation is an exception as it creates multiple jobs. If
  // that's the case, we proceed with the first job. If caller needs particular
  // CUDA job, it should be controlled via --cuda-{host|device}-only option
  // passed to the driver.
  const driver::JobList &Jobs = C->getJobs();
  bool CudaCompilation = false;
  if (Jobs.size() > 1) {
    for (auto &A : C->getActions()){
      // On MacOSX real actions may end up being wrapped in BindArchAction
      if (isa<driver::BindArchAction>(A))
        A = *A->begin();
      if (isa<driver::CudaDeviceAction>(A)) {
        CudaCompilation = true;
        break;
      }
    }
  }
  if (Jobs.size() == 0 || !isa<driver::Command>(*Jobs.begin()) ||
      (Jobs.size() > 1 && !CudaCompilation)) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    Jobs.Print(OS, "; ", true);
    Diags->Report(diag::err_fe_expected_compiler_job) << OS.str();
    return nullptr;
  }

  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
  if (StringRef(Cmd.getCreator().getName()) != "clang") {
    Diags->Report(diag::err_fe_expected_clang_command);
    return nullptr;
  }

  const ArgStringList &CCArgs = Cmd.getArguments();
  std::unique_ptr<CompilerInvocation> CI(new CompilerInvocation());
  if (!CompilerInvocation::CreateFromArgs(*CI,
                                     const_cast<const char **>(CCArgs.data()),
                                     const_cast<const char **>(CCArgs.data()) +
                                     CCArgs.size(),
                                     *Diags))
    return nullptr;
  return CI.release();
}
