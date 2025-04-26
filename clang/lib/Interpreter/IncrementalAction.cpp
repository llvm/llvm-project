//===--- IncrementalAction.h - Incremental Frontend Action -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncrementalAction.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
IncrementalAction::IncrementalAction(CompilerInstance &CI,
                                     llvm::LLVMContext &LLVMCtx,
                                     llvm::Error &Err, Interpreter &I,
                                     std::unique_ptr<ASTConsumer> Consumer)
    : WrapperFrontendAction([&]() {
        llvm::ErrorAsOutParameter EAO(&Err);
        std::unique_ptr<FrontendAction> Act;
        switch (CI.getFrontendOpts().ProgramAction) {
        default:
          Err = llvm::createStringError(
              std::errc::state_not_recoverable,
              "Driver initialization failed. "
              "Incremental mode for action %d is not supported",
              CI.getFrontendOpts().ProgramAction);
          return Act;
        case frontend::ASTDump:
        case frontend::ASTPrint:
        case frontend::ParseSyntaxOnly:
          Act = CreateFrontendAction(CI);
          break;
        case frontend::PluginAction:
        case frontend::EmitAssembly:
        case frontend::EmitBC:
        case frontend::EmitObj:
        case frontend::PrintPreprocessedInput:
        case frontend::EmitLLVMOnly:
          Act.reset(new EmitLLVMOnlyAction(&LLVMCtx));
          break;
        }
        return Act;
      }()),
      Interp(I), Consumer(std::move(Consumer)) {}

std::unique_ptr<ASTConsumer>
IncrementalAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<ASTConsumer> C =
      WrapperFrontendAction::CreateASTConsumer(CI, InFile);

  if (Consumer) {
    std::vector<std::unique_ptr<ASTConsumer>> Cs;
    Cs.push_back(std::move(Consumer));
    Cs.push_back(std::move(C));
    return std::make_unique<MultiplexConsumer>(std::move(Cs));
  }

  return std::make_unique<InProcessPrintingASTConsumer>(std::move(C), Interp);
}

void IncrementalAction::ExecuteAction() {
  WrapperFrontendAction::ExecuteAction();
  getCompilerInstance().getSema().CurContext = nullptr;
}

void IncrementalAction::EndSourceFile() {
  if (IsTerminating && getWrapped())
    WrapperFrontendAction::EndSourceFile();
}

void IncrementalAction::FinalizeAction() {
  assert(!IsTerminating && "Already finalized!");
  IsTerminating = true;
  EndSourceFile();
}

InProcessPrintingASTConsumer::InProcessPrintingASTConsumer(
    std::unique_ptr<ASTConsumer> C, Interpreter &I)
    : MultiplexConsumer(std::move(C)), Interp(I) {}

bool InProcessPrintingASTConsumer::HandleTopLevelDecl(DeclGroupRef DGR) {
  if (DGR.isNull())
    return true;

  for (Decl *D : DGR)
    if (auto *TLSD = llvm::dyn_cast<TopLevelStmtDecl>(D))
      if (TLSD && TLSD->isSemiMissing()) {
        auto ExprOrErr =
            Interp.ExtractValueFromExpr(cast<Expr>(TLSD->getStmt()));
        if (llvm::Error E = ExprOrErr.takeError()) {
          llvm::logAllUnhandledErrors(std::move(E), llvm::errs(),
                                      "Value printing failed: ");
          return false; // abort parsing
        }
        TLSD->setStmt(*ExprOrErr);
      }

  return MultiplexConsumer::HandleTopLevelDecl(DGR);
}

} // namespace clang