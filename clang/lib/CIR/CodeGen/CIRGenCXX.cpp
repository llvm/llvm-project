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

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

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

static void buildDeclDestory(CIRGenFunction &CGF, const VarDecl *D,
                             Address DeclPtr) {
  // Honor __attribute__((no_destroy)) and bail instead of attempting
  // to emit a reference to a possibly nonexistent destructor, which
  // in turn can cause a crash. This will result in a global constructor
  // that isn't balanced out by a destructor call as intended by the
  // attribute. This also checks for -fno-c++-static-destructors and
  // bails even if the attribute is not present.
  assert(D->needsDestruction(CGF.getContext()) == QualType::DK_cxx_destructor);

  auto &CGM = CGF.CGM;

  // If __cxa_atexit is disabled via a flag, a different helper function is
  // generated elsewhere which uses atexit instead, and it takes the destructor
  // directly.
  auto UsingExternalHelper = CGM.getCodeGenOpts().CXAAtExit;
  QualType type = D->getType();
  const CXXRecordDecl *Record = type->getAsCXXRecordDecl();
  bool CanRegisterDestructor =
      Record && (!CGM.getCXXABI().HasThisReturn(
                     GlobalDecl(Record->getDestructor(), Dtor_Complete)) ||
                 CGM.getCXXABI().canCallMismatchedFunctionType());
  if (Record && (CanRegisterDestructor || UsingExternalHelper)) {
    assert(!D->getTLSKind() && "TLS NYI");
    CXXDestructorDecl *Dtor = Record->getDestructor();
    CGM.getCXXABI().buildDestructorCall(CGF, Dtor, Dtor_Complete,
                                        /*ForVirtualBase=*/false,
                                        /*Delegating=*/false, DeclPtr, type);
  } else {
    llvm_unreachable("array destructors not yet supported!");
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
                                                mlir::cir::GlobalOp Addr,
                                                bool NeedsCtor,
                                                bool NeedsDtor) {
  assert(D && " Expected a global declaration!");
  CIRGenFunction CGF{*this, builder, true};
  CurCGF = &CGF;
  CurCGF->CurFn = Addr;
  Addr.setAstAttr(mlir::cir::ASTVarDeclAttr::get(builder.getContext(), D));

  if (NeedsCtor) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto block = builder.createBlock(&Addr.getCtorRegion());
    builder.setInsertionPointToStart(block);
    Address DeclAddr(getAddrOfGlobalVar(D), getASTContext().getDeclAlign(D));
    buildDeclInit(CGF, D, DeclAddr);
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::cir::YieldOp>(Addr->getLoc());
  }

  if (NeedsDtor) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto block = builder.createBlock(&Addr.getDtorRegion());
    builder.setInsertionPointToStart(block);
    Address DeclAddr(getAddrOfGlobalVar(D), getASTContext().getDeclAlign(D));
    buildDeclDestory(CGF, D, DeclAddr);
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::cir::YieldOp>(Addr->getLoc());
  }

  CurCGF = nullptr;
}
