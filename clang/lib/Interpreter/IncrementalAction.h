//===--- IncrementalAction.h - Incremental Frontend Action -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_INCREMENTALACTION_H
#define LLVM_CLANG_INTERPRETER_INCREMENTALACTION_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {
class LLVMContext;
class Module;
}

namespace clang {

class Interpreter;
class CodeGenerator;

/// Records the implicit instantiations of the current input. If the input
/// fails, \p IncrementalParser resets them, and the next use instantiates them
/// again.
class ImplicitInstantiationRecorder : public ASTConsumer,
                                      public ASTMutationListener {
public:
  /// Specializations that got a point of instantiation.
  llvm::SmallSetVector<ValueDecl *, 8> Requested;
  /// Functions that got an instantiated definition.
  llvm::SmallSetVector<FunctionDecl *, 8> Functions;
  /// Variables that got an instantiated definition or initializer.
  llvm::SmallSetVector<VarDecl *, 8> Variables;
  /// Functions whose return type was deduced from the instantiated definition.
  llvm::SmallPtrSet<const FunctionDecl *, 8> DeducedReturnTypes;

  ASTMutationListener *GetASTMutationListener() override { return this; }

  // The listener gets const decls. The reset changes them.
  void InstantiationRequested(const ValueDecl *D) override {
    Requested.insert(const_cast<ValueDecl *>(D));
  }
  void FunctionDefinitionInstantiated(const FunctionDecl *D) override {
    Functions.insert(const_cast<FunctionDecl *>(D));
  }
  void VariableDefinitionInstantiated(const VarDecl *D) override {
    Variables.insert(const_cast<VarDecl *>(D));
  }
  void DeducedReturnType(const FunctionDecl *FD, QualType ReturnType) override {
    DeducedReturnTypes.insert(FD);
  }

  void clear() {
    Requested.clear();
    Functions.clear();
    Variables.clear();
    DeducedReturnTypes.clear();
  }
};

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
  [[maybe_unused]] CompilerInstance &CI;
  std::unique_ptr<ASTConsumer> Consumer;

  /// Owned by the consumer chain of the CompilerInstance.
  ImplicitInstantiationRecorder *Instantiations = nullptr;

  /// When CodeGen is created the first llvm::Module gets cached in many places
  /// and we must keep it alive.
  std::unique_ptr<llvm::Module> CachedInCodeGenModule;

public:
  IncrementalAction(CompilerInstance &Instance, llvm::LLVMContext &LLVMCtx,
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

  /// Cache the current CodeGen module to preserve internal references.
  void CacheCodeGenModule();

  /// Access the cached CodeGen module.
  llvm::Module *getCachedCodeGenModule() const;

  /// Access the current code generator.
  CodeGenerator *getCodeGen() const;

  /// The implicit instantiations of the current input.
  ImplicitInstantiationRecorder &getImplicitInstantiations() {
    return *Instantiations;
  }

  /// Generate an LLVM module for the most recent parsed input.
  std::unique_ptr<llvm::Module> GenModule();
};

class InProcessPrintingASTConsumer final : public MultiplexConsumer {
  Interpreter &Interp;

public:
  InProcessPrintingASTConsumer(std::vector<std::unique_ptr<ASTConsumer>> Cs,
                               Interpreter &I);

  bool HandleTopLevelDecl(DeclGroupRef DGR) override;
};

} // end namespace clang

#endif // LLVM_CLANG_INTERPRETER_INCREMENTALACTION_H
