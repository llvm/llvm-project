//===--- CIRGenerator.cpp - Emit CIR from ASTs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to CIR.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "clang/AST/DeclGroup.h"
#include "clang/CIR/CIRGenerator.h"

using namespace cir;
using namespace clang;

void CIRGenerator::anchor() {}

CIRGenerator::CIRGenerator(clang::DiagnosticsEngine &diags,
                           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                           const CodeGenOptions &CGO)
    : Diags(diags), fs(std::move(vfs)), codeGenOpts{CGO},
      HandlingTopLevelDecls(0) {}
CIRGenerator::~CIRGenerator() {}

void CIRGenerator::Initialize(ASTContext &astCtx) {
  using namespace llvm;

  this->astCtx = &astCtx;

  CGM = std::make_unique<CIRGenModule>(*mlirCtx.get(), astCtx, codeGenOpts,
                                       Diags);
}

bool CIRGenerator::HandleTopLevelDecl(DeclGroupRef D) {

  for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I) {
    CGM->buildTopLevelDecl(*I);
  }

  return true;
}
