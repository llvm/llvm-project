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
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
IncrementalAction::IncrementalAction(CompilerInstance &Instance,
                                     llvm::LLVMContext &LLVMCtx,
                                     llvm::Error &Err, Interpreter &I,
                                     std::unique_ptr<ASTConsumer> Consumer)
    : WrapperFrontendAction([&]() {
        llvm::ErrorAsOutParameter EAO(&Err);
        std::unique_ptr<FrontendAction> Act;
        switch (Instance.getFrontendOpts().ProgramAction) {
        default:
          Err = llvm::createStringError(
              std::errc::state_not_recoverable,
              "Driver initialization failed. "
              "Incremental mode for action %d is not supported",
              Instance.getFrontendOpts().ProgramAction);
          return Act;
        case frontend::ASTDump:
        case frontend::ASTPrint:
        case frontend::ParseSyntaxOnly:
          Act = CreateFrontendAction(Instance);
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
      Interp(I), CI(Instance), Consumer(std::move(Consumer)) {}

std::unique_ptr<ASTConsumer>
IncrementalAction::CreateASTConsumer(CompilerInstance & /*CI*/,
                                     StringRef InFile) {
  std::unique_ptr<ASTConsumer> C =
      WrapperFrontendAction::CreateASTConsumer(this->CI, InFile);

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

void IncrementalAction::CacheCodeGenModule() {
  CachedInCodeGenModule = GenModule();
}

llvm::Module *IncrementalAction::getCachedCodeGenModule() const {
  return CachedInCodeGenModule.get();
}

std::unique_ptr<llvm::Module> IncrementalAction::GenModule() {
  static unsigned ID = 0;
  if (CodeGenerator *CG = getCodeGen()) {
    // Clang's CodeGen is designed to work with a single llvm::Module. In many
    // cases for convenience various CodeGen parts have a reference to the
    // llvm::Module (TheModule or Module) which does not change when a new
    // module is pushed. However, the execution engine wants to take ownership
    // of the module which does not map well to CodeGen's design. To work this
    // around we created an empty module to make CodeGen happy. We should make
    // sure it always stays empty.
    assert(((!CachedInCodeGenModule ||
             !CI.getPreprocessorOpts().Includes.empty()) ||
            (CachedInCodeGenModule->empty() &&
             CachedInCodeGenModule->global_empty() &&
             CachedInCodeGenModule->alias_empty() &&
             CachedInCodeGenModule->ifunc_empty())) &&
           "CodeGen wrote to a readonly module");
    std::unique_ptr<llvm::Module> M(CG->ReleaseModule());
    CG->StartModule("incr_module_" + std::to_string(ID++), M->getContext());
    return M;
  }
  return nullptr;
}

CodeGenerator *IncrementalAction::getCodeGen() const {
  FrontendAction *WrappedAct = getWrapped();
  if (!WrappedAct || !WrappedAct->hasIRSupport())
    return nullptr;
  return static_cast<CodeGenAction *>(WrappedAct)->getCodeGenerator();
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
        auto ExprOrErr = Interp.convertExprToValue(cast<Expr>(TLSD->getStmt()));
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
