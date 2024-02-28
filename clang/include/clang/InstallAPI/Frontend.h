//===- InstallAPI/Frontend.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Top level wrappers for InstallAPI frontend operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_FRONTEND_H
#define LLVM_CLANG_INSTALLAPI_FRONTEND_H

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/InstallAPI/Context.h"
#include "clang/InstallAPI/Visitor.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"

namespace clang {
namespace installapi {

/// Create a buffer that contains all headers to scan
/// for global symbols with.
std::unique_ptr<llvm::MemoryBuffer>
createInputBuffer(const InstallAPIContext &Ctx);

class InstallAPIAction : public ASTFrontendAction {
public:
  explicit InstallAPIAction(llvm::MachO::RecordsSlice &Records)
      : Records(Records) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<InstallAPIVisitor>(CI.getASTContext(), Records);
  }

private:
  llvm::MachO::RecordsSlice &Records;
};
} // namespace installapi
} // namespace clang

#endif // LLVM_CLANG_INSTALLAPI_FRONTEND_H
