//===-- ClangDoc.cpp - ClangDoc ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entry point for the clang-doc tool. It runs
// the clang-doc mapper on a given set of source code files using a
// FrontendActionFactory.
//
//===----------------------------------------------------------------------===//

#include "ClangDoc.h"
#include "Mapper.h"
#include "Representation.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"

namespace clang {
namespace doc {

class MapperActionFactory : public tooling::FrontendActionFactory {
public:
  MapperActionFactory(ClangDocContext CDCtx) : CDCtx(CDCtx) {}
  OwnedPtr<FrontendAction> create() override;

private:
  ClangDocContext CDCtx;
};

OwnedPtr<FrontendAction> MapperActionFactory::create() {
  class ClangDocAction : public clang::ASTFrontendAction {
  public:
    ClangDocAction(ClangDocContext CDCtx) : CDCtx(CDCtx) {}

    OwnedPtr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &Compiler,
                      llvm::StringRef InFile) override {
      return std::make_unique<MapASTVisitor>(&Compiler.getASTContext(), CDCtx);
    }

  private:
    ClangDocContext CDCtx;
  };
  return std::make_unique<ClangDocAction>(CDCtx);
}

OwnedPtr<tooling::FrontendActionFactory>
newMapperActionFactory(ClangDocContext CDCtx) {
  return std::make_unique<MapperActionFactory>(CDCtx);
}

} // namespace doc
} // namespace clang
