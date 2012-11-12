//===--- CodeGenAction.h - LLVM Code Generation Frontend Action -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_CODE_GEN_ACTION_H
#define LLVM_CLANG_CODEGEN_CODE_GEN_ACTION_H

#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/OwningPtr.h"

namespace llvm {
  class LLVMContext;
  class Module;
}

namespace clang {
class BackendConsumer;

class CodeGenAction : public ASTFrontendAction {
private:
  unsigned Act;
  OwningPtr<llvm::Module> TheModule;
  llvm::Module *LinkModule;
  llvm::LLVMContext *VMContext;
  bool OwnsVMContext;

protected:
  /// Create a new code generation action.  If the optional \p _VMContext
  /// parameter is supplied, the action uses it without taking ownership,
  /// otherwise it creates a fresh LLVM context and takes ownership.
  CodeGenAction(unsigned _Act, llvm::LLVMContext *_VMContext = 0);

  virtual bool hasIRSupport() const;

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);

  virtual void ExecuteAction();

  virtual void EndSourceFileAction();

public:
  ~CodeGenAction();

  /// setLinkModule - Set the link module to be used by this action.  If a link
  /// module is not provided, and CodeGenOptions::LinkBitcodeFile is non-empty,
  /// the action will load it from the specified file.
  void setLinkModule(llvm::Module *Mod) { LinkModule = Mod; }

  /// takeModule - Take the generated LLVM module, for use after the action has
  /// been run. The result may be null on failure.
  llvm::Module *takeModule();

  /// Take the LLVM context used by this action.
  llvm::LLVMContext *takeLLVMContext();

  BackendConsumer *BEConsumer;
};

class EmitAssemblyAction : public CodeGenAction {
  virtual void anchor();
public:
  EmitAssemblyAction(llvm::LLVMContext *_VMContext = 0);
};

class EmitBCAction : public CodeGenAction {
  virtual void anchor();
public:
  EmitBCAction(llvm::LLVMContext *_VMContext = 0);
};

class EmitLLVMAction : public CodeGenAction {
  virtual void anchor();
public:
  EmitLLVMAction(llvm::LLVMContext *_VMContext = 0);
};

class EmitLLVMOnlyAction : public CodeGenAction {
  virtual void anchor();
public:
  EmitLLVMOnlyAction(llvm::LLVMContext *_VMContext = 0);
};

class EmitCodeGenOnlyAction : public CodeGenAction {
  virtual void anchor();
public:
  EmitCodeGenOnlyAction(llvm::LLVMContext *_VMContext = 0);
};

class EmitObjAction : public CodeGenAction {
  virtual void anchor();
public:
  EmitObjAction(llvm::LLVMContext *_VMContext = 0);
};

}

#endif
