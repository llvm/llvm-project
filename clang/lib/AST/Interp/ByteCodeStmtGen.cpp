//===--- ByteCodeStmtGen.cpp - Code generator for expressions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ByteCodeStmtGen.h"
#include "ByteCodeEmitter.h"
#include "ByteCodeGenError.h"
#include "Context.h"
#include "Function.h"
#include "PrimType.h"
#include "Program.h"
#include "State.h"
#include "clang/Basic/LLVM.h"

using namespace clang;
using namespace clang::interp;

namespace clang {
namespace interp {

/// Scope managing label targets.
template <class Emitter> class LabelScope {
public:
  virtual ~LabelScope() {  }

protected:
  LabelScope(ByteCodeStmtGen<Emitter> *Ctx) : Ctx(Ctx) {}
  /// ByteCodeStmtGen instance.
  ByteCodeStmtGen<Emitter> *Ctx;
};

/// Sets the context for break/continue statements.
template <class Emitter> class LoopScope final : public LabelScope<Emitter> {
public:
  using LabelTy = typename ByteCodeStmtGen<Emitter>::LabelTy;
  using OptLabelTy = typename ByteCodeStmtGen<Emitter>::OptLabelTy;

  LoopScope(ByteCodeStmtGen<Emitter> *Ctx, LabelTy BreakLabel,
            LabelTy ContinueLabel)
      : LabelScope<Emitter>(Ctx), OldBreakLabel(Ctx->BreakLabel),
        OldContinueLabel(Ctx->ContinueLabel) {
    this->Ctx->BreakLabel = BreakLabel;
    this->Ctx->ContinueLabel = ContinueLabel;
  }

  ~LoopScope() {
    this->Ctx->BreakLabel = OldBreakLabel;
    this->Ctx->ContinueLabel = OldContinueLabel;
  }

private:
  OptLabelTy OldBreakLabel;
  OptLabelTy OldContinueLabel;
};

// Sets the context for a switch scope, mapping labels.
template <class Emitter> class SwitchScope final : public LabelScope<Emitter> {
public:
  using LabelTy = typename ByteCodeStmtGen<Emitter>::LabelTy;
  using OptLabelTy = typename ByteCodeStmtGen<Emitter>::OptLabelTy;
  using CaseMap = typename ByteCodeStmtGen<Emitter>::CaseMap;

  SwitchScope(ByteCodeStmtGen<Emitter> *Ctx, CaseMap &&CaseLabels,
              LabelTy BreakLabel, OptLabelTy DefaultLabel)
      : LabelScope<Emitter>(Ctx), OldBreakLabel(Ctx->BreakLabel),
        OldDefaultLabel(this->Ctx->DefaultLabel),
        OldCaseLabels(std::move(this->Ctx->CaseLabels)) {
    this->Ctx->BreakLabel = BreakLabel;
    this->Ctx->DefaultLabel = DefaultLabel;
    this->Ctx->CaseLabels = std::move(CaseLabels);
  }

  ~SwitchScope() {
    this->Ctx->BreakLabel = OldBreakLabel;
    this->Ctx->DefaultLabel = OldDefaultLabel;
    this->Ctx->CaseLabels = std::move(OldCaseLabels);
  }

private:
  OptLabelTy OldBreakLabel;
  OptLabelTy OldDefaultLabel;
  CaseMap OldCaseLabels;
};

} // namespace interp
} // namespace clang

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitFunc(const FunctionDecl *F) {
  // Classify the return type.
  ReturnType = this->classify(F->getReturnType());

  // Constructor. Set up field initializers.
  if (const auto Ctor = dyn_cast<CXXConstructorDecl>(F)) {
    const RecordDecl *RD = Ctor->getParent();
    const Record *R = this->getRecord(RD);
    if (!R)
      return false;

    for (const auto *Init : Ctor->inits()) {
      const Expr *InitExpr = Init->getInit();
      if (const FieldDecl *Member = Init->getMember()) {
        const Record::Field *F = R->getField(Member);

        if (std::optional<PrimType> T = this->classify(InitExpr)) {
          if (!this->emitThis(InitExpr))
            return false;

          if (!this->visit(InitExpr))
            return false;

          if (!this->emitInitField(*T, F->Offset, InitExpr))
            return false;

          if (!this->emitPopPtr(InitExpr))
            return false;
        } else {
          // Non-primitive case. Get a pointer to the field-to-initialize
          // on the stack and call visitInitialzer() for it.
          if (!this->emitThis(InitExpr))
            return false;

          if (!this->emitGetPtrField(F->Offset, InitExpr))
            return false;

          if (!this->visitInitializer(InitExpr))
            return false;

          if (!this->emitPopPtr(InitExpr))
            return false;
        }
      } else if (const Type *Base = Init->getBaseClass()) {
        // Base class initializer.
        // Get This Base and call initializer on it.
        auto *BaseDecl = Base->getAsCXXRecordDecl();
        assert(BaseDecl);
        const Record::Base *B = R->getBase(BaseDecl);
        assert(B);
        if (!this->emitGetPtrThisBase(B->Offset, InitExpr))
          return false;
        if (!this->visitInitializer(InitExpr))
          return false;
        if (!this->emitPopPtr(InitExpr))
          return false;
      }
    }
  }

  if (const auto *Body = F->getBody())
    if (!visitStmt(Body))
      return false;

  // Emit a guard return to protect against a code path missing one.
  if (F->getReturnType()->isVoidType())
    return this->emitRetVoid(SourceInfo{});
  else
    return this->emitNoRet(SourceInfo{});
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitStmt(const Stmt *S) {
  switch (S->getStmtClass()) {
  case Stmt::CompoundStmtClass:
    return visitCompoundStmt(cast<CompoundStmt>(S));
  case Stmt::DeclStmtClass:
    return visitDeclStmt(cast<DeclStmt>(S));
  case Stmt::ReturnStmtClass:
    return visitReturnStmt(cast<ReturnStmt>(S));
  case Stmt::IfStmtClass:
    return visitIfStmt(cast<IfStmt>(S));
  case Stmt::WhileStmtClass:
    return visitWhileStmt(cast<WhileStmt>(S));
  case Stmt::DoStmtClass:
    return visitDoStmt(cast<DoStmt>(S));
  case Stmt::ForStmtClass:
    return visitForStmt(cast<ForStmt>(S));
  case Stmt::BreakStmtClass:
    return visitBreakStmt(cast<BreakStmt>(S));
  case Stmt::ContinueStmtClass:
    return visitContinueStmt(cast<ContinueStmt>(S));
  case Stmt::SwitchStmtClass:
    return visitSwitchStmt(cast<SwitchStmt>(S));
  case Stmt::CaseStmtClass:
    return visitCaseStmt(cast<CaseStmt>(S));
  case Stmt::DefaultStmtClass:
    return visitDefaultStmt(cast<DefaultStmt>(S));
  case Stmt::NullStmtClass:
    return true;
  default: {
    if (auto *Exp = dyn_cast<Expr>(S))
      return this->discard(Exp);
    return this->bail(S);
  }
  }
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitCompoundStmt(
    const CompoundStmt *CompoundStmt) {
  BlockScope<Emitter> Scope(this);
  for (auto *InnerStmt : CompoundStmt->body())
    if (!visitStmt(InnerStmt))
      return false;
  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitDeclStmt(const DeclStmt *DS) {
  for (auto *D : DS->decls()) {
    // Variable declarator.
    if (auto *VD = dyn_cast<VarDecl>(D)) {
      if (!this->visitVarDecl(VD))
        return false;
      continue;
    }

    // Decomposition declarator.
    if (auto *DD = dyn_cast<DecompositionDecl>(D)) {
      return this->bail(DD);
    }
  }

  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitReturnStmt(const ReturnStmt *RS) {
  if (const Expr *RE = RS->getRetValue()) {
    ExprScope<Emitter> RetScope(this);
    if (ReturnType) {
      // Primitive types are simply returned.
      if (!this->visit(RE))
        return false;
      this->emitCleanup();
      return this->emitRet(*ReturnType, RS);
    } else {
      // RVO - construct the value in the return location.
      if (!this->emitRVOPtr(RE))
        return false;
      if (!this->visitInitializer(RE))
        return false;
      if (!this->emitPopPtr(RE))
        return false;

      this->emitCleanup();
      return this->emitRetVoid(RS);
    }
  }

  // Void return.
  this->emitCleanup();
  return this->emitRetVoid(RS);
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitIfStmt(const IfStmt *IS) {
  BlockScope<Emitter> IfScope(this);

  if (IS->isNonNegatedConsteval())
    return visitStmt(IS->getThen());
  if (IS->isNegatedConsteval())
    return IS->getElse() ? visitStmt(IS->getElse()) : true;

  if (auto *CondInit = IS->getInit())
    if (!visitStmt(IS->getInit()))
      return false;

  if (const DeclStmt *CondDecl = IS->getConditionVariableDeclStmt())
    if (!visitDeclStmt(CondDecl))
      return false;

  if (!this->visitBool(IS->getCond()))
    return false;

  if (const Stmt *Else = IS->getElse()) {
    LabelTy LabelElse = this->getLabel();
    LabelTy LabelEnd = this->getLabel();
    if (!this->jumpFalse(LabelElse))
      return false;
    if (!visitStmt(IS->getThen()))
      return false;
    if (!this->jump(LabelEnd))
      return false;
    this->emitLabel(LabelElse);
    if (!visitStmt(Else))
      return false;
    this->emitLabel(LabelEnd);
  } else {
    LabelTy LabelEnd = this->getLabel();
    if (!this->jumpFalse(LabelEnd))
      return false;
    if (!visitStmt(IS->getThen()))
      return false;
    this->emitLabel(LabelEnd);
  }

  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitWhileStmt(const WhileStmt *S) {
  const Expr *Cond = S->getCond();
  const Stmt *Body = S->getBody();

  LabelTy CondLabel = this->getLabel(); // Label before the condition.
  LabelTy EndLabel = this->getLabel();  // Label after the loop.
  LoopScope<Emitter> LS(this, EndLabel, CondLabel);

  this->emitLabel(CondLabel);
  if (!this->visitBool(Cond))
    return false;
  if (!this->jumpFalse(EndLabel))
    return false;

  if (!this->visitStmt(Body))
    return false;
  if (!this->jump(CondLabel))
    return false;

  this->emitLabel(EndLabel);

  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitDoStmt(const DoStmt *S) {
  const Expr *Cond = S->getCond();
  const Stmt *Body = S->getBody();

  LabelTy StartLabel = this->getLabel();
  LabelTy EndLabel = this->getLabel();
  LabelTy CondLabel = this->getLabel();
  LoopScope<Emitter> LS(this, EndLabel, CondLabel);

  this->emitLabel(StartLabel);
  if (!this->visitStmt(Body))
    return false;
  this->emitLabel(CondLabel);
  if (!this->visitBool(Cond))
    return false;
  if (!this->jumpTrue(StartLabel))
    return false;
  this->emitLabel(EndLabel);
  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitForStmt(const ForStmt *S) {
  // for (Init; Cond; Inc) { Body }
  const Stmt *Init = S->getInit();
  const Expr *Cond = S->getCond();
  const Expr *Inc = S->getInc();
  const Stmt *Body = S->getBody();

  LabelTy EndLabel = this->getLabel();
  LabelTy CondLabel = this->getLabel();
  LabelTy IncLabel = this->getLabel();
  LoopScope<Emitter> LS(this, EndLabel, IncLabel);

  if (Init && !this->visitStmt(Init))
    return false;
  this->emitLabel(CondLabel);
  if (Cond) {
    if (!this->visitBool(Cond))
      return false;
    if (!this->jumpFalse(EndLabel))
      return false;
  }
  if (Body && !this->visitStmt(Body))
    return false;
  this->emitLabel(IncLabel);
  if (Inc && !this->discard(Inc))
    return false;
  if (!this->jump(CondLabel))
    return false;
  this->emitLabel(EndLabel);
  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitBreakStmt(const BreakStmt *S) {
  if (!BreakLabel)
    return false;

  return this->jump(*BreakLabel);
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitContinueStmt(const ContinueStmt *S) {
  if (!ContinueLabel)
    return false;

  return this->jump(*ContinueLabel);
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitSwitchStmt(const SwitchStmt *S) {
  const Expr *Cond = S->getCond();
  PrimType CondT = this->classifyPrim(Cond->getType());

  LabelTy EndLabel = this->getLabel();
  OptLabelTy DefaultLabel = std::nullopt;
  unsigned CondVar = this->allocateLocalPrimitive(Cond, CondT, true, false);

  if (const auto *CondInit = S->getInit())
    if (!visitStmt(CondInit))
      return false;

  // Initialize condition variable.
  if (!this->visit(Cond))
    return false;
  if (!this->emitSetLocal(CondT, CondVar, S))
    return false;

  CaseMap CaseLabels;
  // Create labels and comparison ops for all case statements.
  for (const SwitchCase *SC = S->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    if (const auto *CS = dyn_cast<CaseStmt>(SC)) {
      // FIXME: Implement ranges.
      if (CS->caseStmtIsGNURange())
        return false;
      CaseLabels[SC] = this->getLabel();

      const Expr *Value = CS->getLHS();
      PrimType ValueT = this->classifyPrim(Value->getType());

      // Compare the case statment's value to the switch condition.
      if (!this->emitGetLocal(CondT, CondVar, CS))
        return false;
      if (!this->visit(Value))
        return false;

      // Compare and jump to the case label.
      if (!this->emitEQ(ValueT, S))
        return false;
      if (!this->jumpTrue(CaseLabels[CS]))
        return false;
    } else {
      assert(!DefaultLabel);
      DefaultLabel = this->getLabel();
    }
  }

  // If none of the conditions above were true, fall through to the default
  // statement or jump after the switch statement.
  if (DefaultLabel) {
    if (!this->jump(*DefaultLabel))
      return false;
  } else {
    if (!this->jump(EndLabel))
      return false;
  }

  SwitchScope<Emitter> SS(this, std::move(CaseLabels), EndLabel, DefaultLabel);
  if (!this->visitStmt(S->getBody()))
    return false;
  this->emitLabel(EndLabel);
  return true;
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitCaseStmt(const CaseStmt *S) {
  this->emitLabel(CaseLabels[S]);
  return this->visitStmt(S->getSubStmt());
}

template <class Emitter>
bool ByteCodeStmtGen<Emitter>::visitDefaultStmt(const DefaultStmt *S) {
  this->emitLabel(*DefaultLabel);
  return this->visitStmt(S->getSubStmt());
}

namespace clang {
namespace interp {

template class ByteCodeStmtGen<ByteCodeEmitter>;

} // namespace interp
} // namespace clang
