//===--- ARCMTActions.cpp - ARC Migrate Tool Frontend Actions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ARCMigrate/ARCMTActions.h"
#include "clang/ARCMigrate/ARCMT.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang;
using namespace arcmt;

bool CheckAction::BeginInvocation(CompilerInstance &CI) {
  if (arcmt::checkForManualIssues(CI.getInvocation(), getCurrentInput(),
                                  CI.getPCHContainerOperations(),
                                  CI.getDiagnostics().getClient()))
    return false; // errors, stop the action.

  // We only want to see warnings reported from arcmt::checkForManualIssues.
  CI.getDiagnostics().setIgnoreAllWarnings(true);
  return true;
}

CheckAction::CheckAction(FrontendAction *WrappedAction)
  : WrapperFrontendAction(WrappedAction) {}

bool ModifyAction::BeginInvocation(CompilerInstance &CI) {
  return !arcmt::applyTransformations(CI.getInvocation(), getCurrentInput(),
                                      CI.getPCHContainerOperations(),
                                      CI.getDiagnostics().getClient());
}

ModifyAction::ModifyAction(FrontendAction *WrappedAction)
  : WrapperFrontendAction(WrappedAction) {}

bool MigrateAction::BeginInvocation(CompilerInstance &CI) {
  if (arcmt::migrateWithTemporaryFiles(
          CI.getInvocation(), getCurrentInput(), CI.getPCHContainerOperations(),
          CI.getDiagnostics().getClient(), MigrateDir, EmitPremigrationARCErros,
          PlistOut))
    return false; // errors, stop the action.

  // We only want to see diagnostics emitted by migrateWithTemporaryFiles.
  CI.getDiagnostics().setIgnoreAllWarnings(true);
  return true;
}

MigrateAction::MigrateAction(FrontendAction *WrappedAction,
                             StringRef migrateDir,
                             StringRef plistOut,
                             bool emitPremigrationARCErrors)
  : WrapperFrontendAction(WrappedAction), MigrateDir(migrateDir),
    PlistOut(plistOut), EmitPremigrationARCErros(emitPremigrationARCErrors) {
  if (MigrateDir.empty())
    MigrateDir = "."; // user current directory if none is given.
}
