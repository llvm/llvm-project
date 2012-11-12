//===-- FrontendActions.h - Useful Frontend Actions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITE_FRONTENDACTIONS_H
#define LLVM_CLANG_REWRITE_FRONTENDACTIONS_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {
class FixItRewriter;
class FixItOptions;

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

class HTMLPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class FixItAction : public ASTFrontendAction {
protected:
  OwningPtr<FixItRewriter> Rewriter;
  OwningPtr<FixItOptions> FixItOpts;

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);

  virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                     StringRef Filename);

  virtual void EndSourceFileAction();

  virtual bool hasASTFileSupport() const { return false; }

public:
  FixItAction();
  ~FixItAction();
};

/// \brief Emits changes to temporary files and uses them for the original
/// frontend action.
class FixItRecompile : public WrapperFrontendAction {
public:
  FixItRecompile(FrontendAction *WrappedAction)
    : WrapperFrontendAction(WrappedAction) {}

protected:
  virtual bool BeginInvocation(CompilerInstance &CI);
};

class RewriteObjCAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class RewriteMacrosAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class RewriteTestAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class RewriteIncludesAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

}  // end namespace clang

#endif
