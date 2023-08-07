//===--- CGCXX.cpp - Emit LLVM Code for declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"

using namespace clang;
using namespace cir;

static void buildDeclInit(CIRGenFunction &CGF, const VarDecl *D,
                          Address DeclPtr) {
  assert((D->hasGlobalStorage() ||
          (D->hasLocalStorage() &&
           CGF.getContext().getLangOpts().OpenCLCPlusPlus)) &&
         "VarDecl must have global or local (in the case of OpenCL) storage!");
  assert(!D->getType()->isReferenceType() &&
         "Should not call buildDeclInit on a reference!");

  QualType type = D->getType();
  LValue lv = CGF.makeAddrLValue(DeclPtr, type);

  const Expr *Init = D->getInit();
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case TEK_Aggregate:
    CGF.buildAggExpr(
        Init, AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                      AggValueSlot::DoesNotNeedGCBarriers,
                                      AggValueSlot::IsNotAliased,
                                      AggValueSlot::DoesNotOverlap));
    return;
  case TEK_Scalar:
    llvm_unreachable("scalar evaluation NYI");
  case TEK_Complex:
    llvm_unreachable("complext evaluation NYI");
  }
}

mlir::cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl GD) {
  const auto &FnInfo = getTypes().arrangeCXXStructorDeclaration(GD);
  auto Fn = getAddrOfCXXStructor(GD, &FnInfo, /*FnType=*/nullptr,
                                 /*DontDefer=*/true, ForDefinition);

  setFunctionLinkage(GD, Fn);
  CIRGenFunction CGF{*this, builder};
  CurCGF = &CGF;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    CGF.generateCode(GD, Fn, FnInfo);
  }
  CurCGF = nullptr;

  // TODO: setNonAliasAttributes
  // TODO: SetLLVMFunctionAttributesForDefinition
  return Fn;
}

void CIRGenModule::codegenGlobalInitCxxStructor(const VarDecl *D,
                                                mlir::cir::GlobalOp Addr) {
  CIRGenFunction CGF{*this, builder, true};
  CurCGF = &CGF;
  CurCGF->CurFn = Addr;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto block = builder.createBlock(&Addr.getCtorRegion());
    builder.setInsertionPointToStart(block);
    Address DeclAddr(getAddrOfGlobalVar(D), getASTContext().getDeclAlign(D));
    buildDeclInit(CGF, D, DeclAddr);
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::cir::YieldOp>(Addr->getLoc());
  }
  CurCGF = nullptr;
}
