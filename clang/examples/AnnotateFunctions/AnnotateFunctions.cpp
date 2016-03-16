//===- AnnotateFunctions.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which adds an annotation to every function.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
using namespace clang;

namespace {

class AnnotateFunctionsConsumer : public ASTConsumer {
public:
  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (auto D : DG)
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
        FD->addAttr(AnnotateAttr::CreateImplicit(FD->getASTContext(),
                                                 "example_annotation"));
    return true;
  }
};

class AnnotateFunctionsAction : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return llvm::make_unique<AnnotateFunctionsConsumer>();
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

}

static FrontendPluginRegistry::Add<AnnotateFunctionsAction>
X("annotate-fns", "annotate functions");
