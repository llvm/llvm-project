//===--- IncrementalAction.h - Incremental Frontend Action -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_INCREMENTALACTION_H
#define LLVM_CLANG_INTERPRETER_INCREMENTALACTION_H

#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"

namespace clang {

class Interpreter;

/// A custom action enabling the incremental processing functionality.
///
/// The usual \p FrontendAction expects one call to ExecuteAction and once it
/// sees a call to \p EndSourceFile it deletes some of the important objects
/// such as \p Preprocessor and \p Sema assuming no further input will come.
///
/// \p IncrementalAction ensures it keep its underlying action's objects alive
/// as long as the \p IncrementalParser needs them.
///
class IncrementalAction : public WrapperFrontendAction {
private:
  bool IsTerminating = false;
  Interpreter &Interp;
  std::unique_ptr<ASTConsumer> Consumer;

public:
  IncrementalAction(CompilerInstance &CI, llvm::LLVMContext &LLVMCtx,
                    llvm::Error &Err, Interpreter &I,
                    std::unique_ptr<ASTConsumer> Consumer = nullptr);

  FrontendAction *getWrapped() const { return WrappedAction.get(); }

  TranslationUnitKind getTranslationUnitKind() override {
    return TU_Incremental;
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

  void ExecuteAction() override;

  // Do not terminate after processing the input. This allows us to keep various
  // clang objects alive and to incrementally grow the current TU.
  void EndSourceFile() override;

  void FinalizeAction();
};

class InProcessPrintingASTConsumer final : public MultiplexConsumer {
  Interpreter &Interp;

public:
  InProcessPrintingASTConsumer(std::unique_ptr<ASTConsumer> C, Interpreter &I);

  bool HandleTopLevelDecl(DeclGroupRef DGR) override;
};

} // end namespace clang

#endif // LLVM_CLANG_INTERPRETER_INCREMENTALACTION_H
