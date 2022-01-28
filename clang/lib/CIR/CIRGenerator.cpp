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

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/CIRGenerator.h"

using namespace cir;

CIRGenerator::CIRGenerator() = default;
CIRGenerator::~CIRGenerator() = default;

void CIRGenerator::Initialize(clang::ASTContext &astCtx) {
  using namespace llvm;

  this->astCtx = &astCtx;

  mlirCtx = std::make_unique<mlir::MLIRContext>();
  mlirCtx->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirCtx->getOrLoadDialect<mlir::cir::CIRDialect>();
  mlirCtx->getOrLoadDialect<mlir::memref::MemRefDialect>();
  CGM = std::make_unique<CIRGenModule>(*mlirCtx.get(), astCtx);
}

void CIRGenerator::verifyModule() { CGM->verifyModule(); }

bool CIRGenerator::EmitFunction(const FunctionDecl *FD) {
  auto func = CGM->buildFunction(FD);
  assert(func && "should emit function");
  return func.getOperation() != nullptr;
}

mlir::ModuleOp CIRGenerator::getModule() { return CGM->getModule(); }

bool CIRGenerator::HandleTopLevelDecl(clang::DeclGroupRef D) {
  for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I) {
    CGM->buildTopLevelDecl(*I);
  }

  return true;
}

void CIRGenerator::HandleTranslationUnit(ASTContext &C) {}
