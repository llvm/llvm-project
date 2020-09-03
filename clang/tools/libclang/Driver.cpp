//===- Driver.cpp - A C Interface for the Clang Driver --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a C API for extracting information from the clang driver.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Driver.h"

#include "CIndexDiagnostic.h"

#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Host.h"

using namespace clang;

class CXDiagnosticSetDiagnosticConsumer : public DiagnosticConsumer {
  SmallVector<StoredDiagnostic, 4> Errors;
public:

  void HandleDiagnostic(DiagnosticsEngine::Level level,
  const Diagnostic &Info) override {
    if (level >= DiagnosticsEngine::Error)
      Errors.push_back(StoredDiagnostic(level, Info));
  }

  CXDiagnosticSet getDiagnosticSet() {
    return cxdiag::createStoredDiags(Errors, LangOptions());
  }
};

CXExternalActionList *
clang_Driver_getExternalActionsForCommand_v0(int ArgC, const char **ArgV,
                                             const char **Environment,
                                             const char *WorkingDirectory,
                                             CXDiagnosticSet *OutDiags) {
  if (OutDiags)
    *OutDiags = nullptr;

  // Non empty environments are not currently supported.
  if (Environment)
    return nullptr;

  // ArgV must at least include the compiler executable name.
  if (ArgC < 1)
    return nullptr;

  CXDiagnosticSetDiagnosticConsumer DiagConsumer;
  auto Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                                   &DiagConsumer, false);

  // Use createPhysicalFileSystem instead of getRealFileSystem so that
  // setCurrentWorkingDirectory doesn't change the working directory of the
  // process.
  std::unique_ptr<llvm::vfs::FileSystem> VFS =
      llvm::vfs::createPhysicalFileSystem();
  if (WorkingDirectory)
    if (std::error_code EC =
          VFS->setCurrentWorkingDirectory(WorkingDirectory)) {
      Diags->Report(diag::err_drv_unable_to_set_working_directory) <<
          WorkingDirectory;
      if (OutDiags)
        *OutDiags = DiagConsumer.getDiagnosticSet();
      return nullptr;
    }

  driver::Driver TheDriver(ArgV[0], llvm::sys::getDefaultTargetTriple(), *Diags,
                           VFS.release());
  TheDriver.setCheckInputsExist(false);
  std::unique_ptr<driver::Compilation> C(
      TheDriver.BuildCompilation(llvm::makeArrayRef(ArgV, ArgC)));
  if (!C) {
    if (OutDiags)
      *OutDiags = DiagConsumer.getDiagnosticSet();
    return nullptr;
  }

  const driver::JobList &Jobs = C->getJobs();
  CXExternalAction **Actions = new CXExternalAction *[Jobs.size()];
  int AI = 0;
  for (auto &&J : Jobs) {
    // First calculate the total space we'll need for this action's arguments.
    llvm::opt::ArgStringList Args = J.getArguments();
    Args.insert(Args.begin(), J.getExecutable());
    int ArgSpace = (Args.size() + 1) * sizeof(const char *);
    for (auto &&Arg : Args)
      ArgSpace += strlen(Arg) + 1; // Null terminator

    // Tail allocate the space for the strings.
    auto Action =
        new ((CXExternalAction *)malloc(sizeof(CXExternalAction) + ArgSpace))
            CXExternalAction;
    Action->ArgC = Args.size();
    Action->ArgV = reinterpret_cast<const char **>(Action + 1);
    Action->ArgV[Args.size()] = nullptr;
    char *StrTable = ((char *)Action) + sizeof(CXExternalAction) +
                     (Args.size() + 1) * sizeof(const char *);
    int I = 0;
    for (auto &&Arg : Args) {
      Action->ArgV[I++] = strcpy(StrTable, Arg);
      StrTable += strlen(Arg) + 1;
    }
    Actions[AI++] = Action;
  }

  return new CXExternalActionList{(int)Jobs.size(), Actions};
}

void clang_Driver_ExternalActionList_dispose(CXExternalActionList *EAL) {
  if (!EAL)
    return;

  for (int I = 0; I < EAL->Count; ++I)
    free(EAL->Actions[I]);
  delete[] EAL->Actions;
  delete EAL;
}
