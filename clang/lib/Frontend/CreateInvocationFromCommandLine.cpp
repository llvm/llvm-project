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

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "llvm/Support/Host.h"
using namespace clang;

/// createInvocationFromCommandLine - Construct a compiler invocation object for
/// a command line argument vector.
///
/// \return A CompilerInvocation, or 0 if none was built for the given
/// argument vector.
CompilerInvocation *
clang::createInvocationFromCommandLine(ArrayRef<const char *> ArgList,
                            IntrusiveRefCntPtr<DiagnosticsEngine> Diags) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                                ArgList.size(),
                                                ArgList.begin());
  }

  SmallVector<const char *, 16> Args;
  Args.push_back("<clang>"); // FIXME: Remove dummy argument.
  Args.insert(Args.end(), ArgList.begin(), ArgList.end());

  // FIXME: Find a cleaner way to force the driver into restricted modes.
  Args.push_back("-fsyntax-only");

  // FIXME: We shouldn't have to pass in the path info.
  driver::Driver TheDriver("clang", llvm::sys::getDefaultTargetTriple(),
                           "a.out", false, *Diags);

  // Don't check that inputs exist, they may have been remapped.
  TheDriver.setCheckInputsExist(false);

  OwningPtr<driver::Compilation> C(TheDriver.BuildCompilation(Args));

  // Just print the cc1 options if -### was present.
  if (C->getArgs().hasArg(driver::options::OPT__HASH_HASH_HASH)) {
    C->PrintJob(llvm::errs(), C->getJobs(), "\n", true);
    return 0;
  }

  // We expect to get back exactly one command job, if we didn't something
  // failed.
  const driver::JobList &Jobs = C->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    C->PrintJob(OS, C->getJobs(), "; ", true);
    Diags->Report(diag::err_fe_expected_compiler_job) << OS.str();
    return 0;
  }

  const driver::Command *Cmd = cast<driver::Command>(*Jobs.begin());
  if (StringRef(Cmd->getCreator().getName()) != "clang") {
    Diags->Report(diag::err_fe_expected_clang_command);
    return 0;
  }

  const driver::ArgStringList &CCArgs = Cmd->getArguments();
  OwningPtr<CompilerInvocation> CI(new CompilerInvocation());
  if (!CompilerInvocation::CreateFromArgs(*CI,
                                     const_cast<const char **>(CCArgs.data()),
                                     const_cast<const char **>(CCArgs.data()) +
                                     CCArgs.size(),
                                     *Diags))
    return 0;
  return CI.take();
}
