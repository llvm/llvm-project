//===--------------- InterpBuiltinConstantP.cpp -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InterpBuiltinConstantP.h"
#include "Compiler.h"
#include "EvalEmitter.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace interp;

bool BCPVisitor::VisitStmt(Stmt *S) {
  switch (S->getStmtClass()) {
  case Stmt::DeclRefExprClass:
    return VisitDeclRefExpr(cast<DeclRefExpr>(S));
  case Stmt::ImplicitCastExprClass:
  case Stmt::CStyleCastExprClass:
    return VisitCastExpr(cast<CastExpr>(S));
  case Stmt::CallExprClass:
    return VisitCallExpr(cast<CallExpr>(S));
  case Stmt::UnaryOperatorClass:
    return VisitUnaryOperator(cast<UnaryOperator>(S));
  case Stmt::BinaryOperatorClass:
    return VisitBinaryOperator(cast<BinaryOperator>(S));
  case Stmt::ParenExprClass:
    return VisitStmt(cast<ParenExpr>(S)->getSubExpr());
  case Stmt::ConditionalOperatorClass:
    return VisitConditionalOperator(cast<ConditionalOperator>(S));
  case Stmt::MemberExprClass:
    return VisitMemberExpr(cast<MemberExpr>(S));

  case Stmt::IntegerLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::CXXNullPtrLiteralExprClass:
    return true;
  default:
    return Fail();
  }

  llvm_unreachable("All handled above");
}

bool BCPVisitor::VisitDeclRefExpr(DeclRefExpr *E) {
  const ValueDecl *D = E->getDecl();

  // Local variable?
  if (auto LocalOffset = findLocal(D)) {
    Result = pointerChainIsLive(Frame->getLocalPointer(*LocalOffset));
  } else if (auto G = S.P.getGlobal(D)) {
    // Fine.
  } else if (auto P = findParam(D)) {
    Result = pointerChainIsLive(Frame->getParamPointer(*P));
  } else {
    Result = false;
  }

  return Result;
}

bool BCPVisitor::VisitUnaryOperator(UnaryOperator *E) {
  switch (E->getOpcode()) {
  case UO_AddrOf:
    Result = isa<CXXTypeidExpr>(E->getSubExpr());
    break;
  case UO_PostInc:
  case UO_PreInc:
  case UO_PostDec:
  case UO_PreDec:
    return Fail();
  default:;
  }
  return Result;
}

bool BCPVisitor::VisitBinaryOperator(BinaryOperator *E) {
  if (E->isCommaOp())
    return VisitStmt(E->getRHS());

  return VisitStmt(E->getLHS()) && VisitStmt(E->getRHS());
}

bool BCPVisitor::VisitCastExpr(CastExpr *E) {
  if (E->getCastKind() == CK_ToVoid)
    return Fail();
  return VisitStmt(E->getSubExpr());
}

bool BCPVisitor::VisitCallExpr(CallExpr *E) {
  // FIXME: We're not passing any arguments to the function call.
  Compiler<EvalEmitter> C(S.getContext(), S.P, S, S.Stk);

  auto OldDiag = S.getEvalStatus().Diag;
  S.getEvalStatus().Diag = nullptr;
  auto Res = C.interpretExpr(E, /*ConvertResultToRValue=*/E->isGLValue());

  S.getEvalStatus().Diag = OldDiag;
  Result = !Res.isInvalid();
  return Result;
}

bool BCPVisitor::VisitConditionalOperator(ConditionalOperator *E) {
  return VisitStmt(E->getCond()) && VisitStmt(E->getTrueExpr()) &&
         VisitStmt(E->getFalseExpr());
}

bool BCPVisitor::VisitMemberExpr(MemberExpr *E) {
  if (!isa<DeclRefExpr>(E->getBase()))
    return Fail();

  const auto *BaseDecl = cast<DeclRefExpr>(E->getBase())->getDecl();
  const FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
  if (!FD)
    return Fail();

  if (!VisitStmt(E->getBase()))
    return Fail();

  Pointer BasePtr = getPointer(BaseDecl);
  const Record *R = BasePtr.getRecord();
  assert(R);
  Pointer FieldPtr = BasePtr.atField(R->getField(FD)->Offset);
  if (!pointerChainIsLive(FieldPtr))
    return Fail();
  return true;
}

std::optional<unsigned> BCPVisitor::findLocal(const ValueDecl *D) const {
  const auto *Func = Frame->getFunction();
  if (!Func)
    return std::nullopt;
  for (auto &Scope : Func->scopes()) {
    for (auto &Local : Scope.locals()) {
      if (Local.Desc->asValueDecl() == D) {
        return Local.Offset;
      }
    }
  }
  return std::nullopt;
}

std::optional<unsigned> BCPVisitor::findParam(const ValueDecl *D) const {
  const auto *Func = Frame->getFunction();
  if (!Func || !Frame->Caller)
    return std::nullopt;

  return Func->findParam(D);
}

bool BCPVisitor::pointerChainIsLive(const Pointer &P) const {
  Pointer Ptr = P;
  for (;;) {
    if (!Ptr.isLive() || !Ptr.isInitialized() || Ptr.isExtern() ||
        Ptr.isDummy())
      return false;

    if (Ptr.isZero())
      return true;

    const Descriptor *Desc = Ptr.getFieldDesc();
    if (!Desc->isPrimitive() || Desc->getPrimType() != PT_Ptr)
      return true;

    Ptr = Ptr.deref<Pointer>();
  }

  return true;
}

Pointer BCPVisitor::getPointer(const ValueDecl *D) const {
  if (auto LocalOffset = findLocal(D))
    return Frame->getLocalPointer(*LocalOffset);
  if (auto G = S.P.getGlobal(D))
    return S.P.getPtrGlobal(*G);
  if (auto P = findParam(D))
    return Frame->getParamPointer(*P);

  llvm_unreachable("One of the ifs before should've worked.");
}
