//===--- ARCMTActions.h - ARC Migrate Tool Frontend Actions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ARCMIGRATE_ARCMTACTIONS_H
#define LLVM_CLANG_ARCMIGRATE_ARCMTACTIONS_H

#include "clang/ARCMigrate/FileRemapper.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Support/Compiler.h"
#include <memory>

namespace clang {
namespace arcmt {

class CLANG_ABI CheckAction : public WrapperFrontendAction {
protected:
  bool BeginInvocation(CompilerInstance &CI) override;

public:
  CheckAction(std::unique_ptr<FrontendAction> WrappedAction);
};

class CLANG_ABI ModifyAction : public WrapperFrontendAction {
protected:
  bool BeginInvocation(CompilerInstance &CI) override;

public:
  ModifyAction(std::unique_ptr<FrontendAction> WrappedAction);
};

class CLANG_ABI MigrateSourceAction : public ASTFrontendAction {
  FileRemapper Remapper;
protected:
  bool BeginInvocation(CompilerInstance &CI) override;
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

class CLANG_ABI MigrateAction : public WrapperFrontendAction {
  std::string MigrateDir;
  std::string PlistOut;
  bool EmitPremigrationARCErrors;
protected:
  bool BeginInvocation(CompilerInstance &CI) override;

public:
  MigrateAction(std::unique_ptr<FrontendAction> WrappedAction,
                StringRef migrateDir,
                StringRef plistOut,
                bool emitPremigrationARCErrors);
};

/// Migrates to modern ObjC syntax.
class CLANG_ABI ObjCMigrateAction : public WrapperFrontendAction {
  std::string MigrateDir;
  unsigned    ObjCMigAction;
  FileRemapper Remapper;
  CompilerInstance *CompInst;
public:
  ObjCMigrateAction(std::unique_ptr<FrontendAction> WrappedAction,
                    StringRef migrateDir, unsigned migrateAction);

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
  bool BeginInvocation(CompilerInstance &CI) override;
};

}
}

#endif
