//===-- FrontendActions.h - Useful Frontend Actions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDACTIONS_H
#define LLVM_CLANG_FRONTEND_FRONTENDACTIONS_H

#include "clang/Frontend/FrontendAction.h"
#include <string>
#include <vector>

namespace clang {

class Module;
  
//===----------------------------------------------------------------------===//
// Custom Consumer Actions
//===----------------------------------------------------------------------===//

class InitOnlyAction : public FrontendAction {
  virtual void ExecuteAction();

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);

public:
  // Don't claim to only use the preprocessor, we want to follow the AST path,
  // but do nothing.
  virtual bool usesPreprocessorOnly() const { return false; }
};

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

class ASTPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class ASTDumpAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class ASTDeclListAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class ASTDumpXMLAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class ASTViewAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class DeclContextPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
};

class GeneratePCHAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);

  virtual TranslationUnitKind getTranslationUnitKind() {
    return TU_Prefix;
  }

  virtual bool hasASTFileSupport() const { return false; }

public:
  /// \brief Compute the AST consumer arguments that will be used to
  /// create the PCHGenerator instance returned by CreateASTConsumer.
  ///
  /// \returns true if an error occurred, false otherwise.
  static bool ComputeASTConsumerArguments(CompilerInstance &CI,
                                          StringRef InFile,
                                          std::string &Sysroot,
                                          std::string &OutputFile,
                                          raw_ostream *&OS);
};

class GenerateModuleAction : public ASTFrontendAction {
  clang::Module *Module;
  
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);
  
  virtual TranslationUnitKind getTranslationUnitKind() { 
    return TU_Module;
  }
  
  virtual bool hasASTFileSupport() const { return false; }
  
public:
  virtual bool BeginSourceFileAction(CompilerInstance &CI, StringRef Filename);
  
  /// \brief Compute the AST consumer arguments that will be used to
  /// create the PCHGenerator instance returned by CreateASTConsumer.
  ///
  /// \returns true if an error occurred, false otherwise.
  static bool ComputeASTConsumerArguments(CompilerInstance &CI,
                                          StringRef InFile,
                                          std::string &Sysroot,
                                          std::string &OutputFile,
                                          raw_ostream *&OS);
};
  
class SyntaxOnlyAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);

public:
  virtual bool hasCodeCompletionSupport() const { return true; }
};

/**
 * \brief Frontend action adaptor that merges ASTs together.
 *
 * This action takes an existing AST file and "merges" it into the AST
 * context, producing a merged context. This action is an action
 * adaptor, which forwards most of its calls to another action that
 * will consume the merged context.
 */
class ASTMergeAction : public FrontendAction {
  /// \brief The action that the merge action adapts.
  FrontendAction *AdaptedAction;
  
  /// \brief The set of AST files to merge.
  std::vector<std::string> ASTFiles;

protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile);

  virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                     StringRef Filename);

  virtual void ExecuteAction();
  virtual void EndSourceFileAction();

public:
  ASTMergeAction(FrontendAction *AdaptedAction, ArrayRef<std::string> ASTFiles);
  virtual ~ASTMergeAction();

  virtual bool usesPreprocessorOnly() const;
  virtual TranslationUnitKind getTranslationUnitKind();
  virtual bool hasPCHSupport() const;
  virtual bool hasASTFileSupport() const;
  virtual bool hasCodeCompletionSupport() const;
};

class PrintPreambleAction : public FrontendAction {
protected:
  void ExecuteAction();
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &, StringRef) { 
    return 0; 
  }
  
  virtual bool usesPreprocessorOnly() const { return true; }
};
  
//===----------------------------------------------------------------------===//
// Preprocessor Actions
//===----------------------------------------------------------------------===//

class DumpRawTokensAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class DumpTokensAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class GeneratePTHAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class PreprocessOnlyAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class PrintPreprocessedAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();

  virtual bool hasPCHSupport() const { return true; }
};
  
}  // end namespace clang

#endif
