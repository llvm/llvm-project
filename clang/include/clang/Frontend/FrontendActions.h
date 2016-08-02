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
class FileEntry;
  
//===----------------------------------------------------------------------===//
// Custom Consumer Actions
//===----------------------------------------------------------------------===//

class InitOnlyAction : public FrontendAction {
  void ExecuteAction() override;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

public:
  // Don't claim to only use the preprocessor, we want to follow the AST path,
  // but do nothing.
  bool usesPreprocessorOnly() const override { return false; }
};

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

class ASTPrintAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

class ASTDumpAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

class ASTDeclListAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

class ASTViewAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

class DeclContextPrintAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

class GeneratePCHAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

  TranslationUnitKind getTranslationUnitKind() override {
    return TU_Prefix;
  }

  bool hasASTFileSupport() const override { return false; }

public:
  /// \brief Compute the AST consumer arguments that will be used to
  /// create the PCHGenerator instance returned by CreateASTConsumer.
  ///
  /// \returns true if an error occurred, false otherwise.
  static raw_pwrite_stream *
  ComputeASTConsumerArguments(CompilerInstance &CI, StringRef InFile,
                              std::string &Sysroot, std::string &OutputFile);
};

class GenerateModuleAction : public ASTFrontendAction {
  clang::Module *Module;
  const FileEntry *ModuleMapForUniquing;
  bool IsSystem;
  
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

  TranslationUnitKind getTranslationUnitKind() override {
    return TU_Module;
  }

  bool hasASTFileSupport() const override { return false; }

public:
  GenerateModuleAction(const FileEntry *ModuleMap = nullptr,
                       bool IsSystem = false)
    : ASTFrontendAction(), ModuleMapForUniquing(ModuleMap), IsSystem(IsSystem)
  { }

  bool BeginSourceFileAction(CompilerInstance &CI, StringRef Filename) override;

  /// \brief Compute the AST consumer arguments that will be used to
  /// create the PCHGenerator instance returned by CreateASTConsumer.
  ///
  /// \returns true if an error occurred, false otherwise.
  raw_pwrite_stream *ComputeASTConsumerArguments(CompilerInstance &CI,
                                                 StringRef InFile,
                                                 std::string &Sysroot,
                                                 std::string &OutputFile);
};

class SyntaxOnlyAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

public:
  bool hasCodeCompletionSupport() const override { return true; }
};

/// \brief Dump information about the given module file, to be used for
/// basic debugging and discovery.
class DumpModuleInfoAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
  void ExecuteAction() override;

public:
  bool hasPCHSupport() const override { return false; }
  bool hasASTFileSupport() const override { return true; }
  bool hasIRSupport() const override { return false; }
  bool hasCodeCompletionSupport() const override { return false; }
};

class VerifyPCHAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

  void ExecuteAction() override;

public:
  bool hasCodeCompletionSupport() const override { return false; }
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
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

  bool BeginSourceFileAction(CompilerInstance &CI,
                             StringRef Filename) override;

  void ExecuteAction() override;
  void EndSourceFileAction() override;

public:
  ASTMergeAction(FrontendAction *AdaptedAction, ArrayRef<std::string> ASTFiles);
  ~ASTMergeAction() override;

  bool usesPreprocessorOnly() const override;
  TranslationUnitKind getTranslationUnitKind() override;
  bool hasPCHSupport() const override;
  bool hasASTFileSupport() const override;
  bool hasCodeCompletionSupport() const override;
};

class PrintPreambleAction : public FrontendAction {
protected:
  void ExecuteAction() override;
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override {
    return nullptr;
  }

  bool usesPreprocessorOnly() const override { return true; }
};
  
//===----------------------------------------------------------------------===//
// Preprocessor Actions
//===----------------------------------------------------------------------===//

class DumpRawTokensAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override;
};

class DumpTokensAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override;
};

class GeneratePTHAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override;
};

class PreprocessOnlyAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override;
};

class PrintPreprocessedAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override;

  bool hasPCHSupport() const override { return true; }
};
  
}  // end namespace clang

#endif
