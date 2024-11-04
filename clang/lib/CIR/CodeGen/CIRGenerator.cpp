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
                           const CodeGenOptions &cgo)
    : diags(diags), fs(std::move(vfs)), codeGenOpts{cgo} {}
CIRGenerator::~CIRGenerator() = default;

void CIRGenerator::Initialize(ASTContext &astCtx) {
  using namespace llvm;

  this->astCtx = &astCtx;

  cgm = std::make_unique<CIRGenModule>(*mlirCtx, astCtx, codeGenOpts, diags);
}

bool CIRGenerator::HandleTopLevelDecl(DeclGroupRef group) {

  for (Decl *decl : group)
    cgm->buildTopLevelDecl(decl);

  return true;
}
