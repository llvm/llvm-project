//===-- ModelInjector.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModelInjector.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/Stack.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include <utility>

using namespace clang;
using namespace ento;

ModelInjector::ModelInjector(CompilerInstance &CI) : CI(CI) {}

Stmt *ModelInjector::getBody(const FunctionDecl *D) {
  onBodySynthesis(D);
  return Bodies[D->getName()];
}

Stmt *ModelInjector::getBody(const ObjCMethodDecl *D) {
  onBodySynthesis(D);
  return Bodies[D->getName()];
}

void ModelInjector::onBodySynthesis(const NamedDecl *D) {

  // FIXME: what about overloads? Declarations can be used as keys but what
  // about file name index? Mangled names may not be suitable for that either.
  if (Bodies.count(D->getName()) != 0)
    return;

  llvm::IntrusiveRefCntPtr<SourceManager> SM = CI.getSourceManagerPtr();
  FileID mainFileID = SM->getMainFileID();

  llvm::StringRef modelPath = CI.getAnalyzerOpts().ModelPath;

  llvm::SmallString<128> fileName;

  if (!modelPath.empty())
    fileName =
        llvm::StringRef(modelPath.str() + "/" + D->getName().str() + ".model");
  else
    fileName = llvm::StringRef(D->getName().str() + ".model");

  if (!llvm::sys::fs::exists(fileName.str())) {
    Bodies[D->getName()] = nullptr;
    return;
  }

  auto Invocation = std::make_shared<CompilerInvocation>(CI.getInvocation());

  FrontendOptions &FrontendOpts = Invocation->getFrontendOpts();
  InputKind IK = Language::CXX; // FIXME
  FrontendOpts.Inputs.clear();
  FrontendOpts.Inputs.emplace_back(fileName, IK);
  FrontendOpts.DisableFree = true;

  Invocation->getDiagnosticOpts().VerifyDiagnostics = 0;

  // Modules are parsed by a separate CompilerInstance, so this code mimics that
  // behavior for models
  CompilerInstance Instance(std::move(Invocation),
                            CI.getPCHContainerOperations());
  Instance.createDiagnostics(
      CI.getVirtualFileSystem(),
      new ForwardingDiagnosticConsumer(CI.getDiagnosticClient()),
      /*ShouldOwnClient=*/true);

  Instance.getDiagnostics().setSourceManager(SM.get());

  // The instance wants to take ownership, however DisableFree frontend option
  // is set to true to avoid double free issues
  Instance.setFileManager(CI.getFileManagerPtr());
  Instance.setSourceManager(SM);
  Instance.setPreprocessor(CI.getPreprocessorPtr());
  Instance.setASTContext(CI.getASTContextPtr());

  Instance.getPreprocessor().InitializeForModelFile();

  ParseModelFileAction parseModelFile(Bodies);

  llvm::CrashRecoveryContext CRC;

  CRC.RunSafelyOnThread([&]() { Instance.ExecuteAction(parseModelFile); },
                        DesiredStackSize);

  Instance.getPreprocessor().FinalizeForModelFile();

  Instance.resetAndLeakSourceManager();
  Instance.resetAndLeakFileManager();
  Instance.resetAndLeakPreprocessor();

  // The preprocessor enters to the main file id when parsing is started, so
  // the main file id is changed to the model file during parsing and it needs
  // to be reset to the former main file id after parsing of the model file
  // is done.
  SM->setMainFileID(mainFileID);
}
