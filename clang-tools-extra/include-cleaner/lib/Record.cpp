//===--- Record.cpp - Record compiler events ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Record.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/SourceManager.h"

namespace clang::include_cleaner {

std::unique_ptr<ASTConsumer> RecordedAST::record() {
  class Recorder : public ASTConsumer {
    RecordedAST *Out;

  public:
    Recorder(RecordedAST *Out) : Out(Out) {}
    void Initialize(ASTContext &Ctx) override { Out->Ctx = &Ctx; }
    bool HandleTopLevelDecl(DeclGroupRef DG) override {
      const auto &SM = Out->Ctx->getSourceManager();
      for (Decl *D : DG) {
        if (!SM.isWrittenInMainFile(SM.getExpansionLoc(D->getLocation())))
          continue;
        // FIXME: Filter out certain Obj-C and template-related decls.
        Out->Roots.push_back(D);
      }
      return ASTConsumer::HandleTopLevelDecl(DG);
    }
  };

  return std::make_unique<Recorder>(this);
}

} // namespace clang::include_cleaner
