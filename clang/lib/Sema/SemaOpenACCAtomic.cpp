//== SemaOpenACCAtomic.cpp - Semantic Analysis for OpenACC Atomic Construct===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for the OpenACC atomic construct.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/SemaOpenACC.h"

#include <optional>

using namespace clang;

namespace {

class AtomicOperandChecker {
  SemaOpenACC &SemaRef;
  OpenACCAtomicKind AtKind;
  SourceLocation AtomicDirLoc;
  StmtResult AssocStmt;

  // Do a diagnostic, which sets the correct error, then displays passed note.
  bool DiagnoseInvalidAtomic(SourceLocation Loc, PartialDiagnostic NoteDiag) {
    SemaRef.Diag(AtomicDirLoc, diag::err_acc_invalid_atomic)
        << (AtKind != OpenACCAtomicKind::None) << AtKind;
    SemaRef.Diag(Loc, NoteDiag);
    return true;
  }

  // Create a replacement recovery expr in case we find an error here.  This
  // allows us to ignore this during template instantiation so we only get a
  // single error.
  StmtResult getRecoveryExpr() {
    if (!AssocStmt.isUsable())
      return AssocStmt;

    if (!SemaRef.getASTContext().getLangOpts().RecoveryAST)
      return StmtError();

    Expr *E = dyn_cast<Expr>(AssocStmt.get());
    QualType T = E ? E->getType() : SemaRef.getASTContext().DependentTy;

    return RecoveryExpr::Create(SemaRef.getASTContext(), T,
                                AssocStmt.get()->getBeginLoc(),
                                AssocStmt.get()->getEndLoc(),
                                E ? ArrayRef<Expr *>{E} : ArrayRef<Expr *>{});
  }

  // OpenACC 3.3 2.12: 'expr' is an expression with scalar type.
  bool CheckOperandExpr(const Expr *E, PartialDiagnostic PD) {
    QualType ExprTy = E->getType();

    // Scalar allowed, plus we allow instantiation dependent to support
    // templates.
    if (ExprTy->isInstantiationDependentType() || ExprTy->isScalarType())
      return false;

    return DiagnoseInvalidAtomic(E->getExprLoc(),
                                 PD << diag::OACCLValScalar::Scalar << ExprTy);
  }

  // OpenACC 3.3 2.12: 'x' and 'v' (as applicable) are boht l-value expressoins
  // with scalar type.
  bool CheckOperandVariable(const Expr *E, PartialDiagnostic PD) {
    if (CheckOperandExpr(E, PD))
      return true;

    if (E->isLValue())
      return false;

    return DiagnoseInvalidAtomic(E->getExprLoc(),
                                 PD << diag::OACCLValScalar::LVal);
  }

  Expr *RequireExpr(Stmt *Stmt, PartialDiagnostic ExpectedNote) {
    if (Expr *E = dyn_cast<Expr>(Stmt))
      return E->IgnoreImpCasts();

    DiagnoseInvalidAtomic(Stmt->getBeginLoc(), ExpectedNote);
    return nullptr;
  }

  // A struct to hold the return the inner components of any operands, which
  // allows for compound checking.
  struct BinaryOpInfo {
    const Expr *FoundExpr = nullptr;
    const Expr *LHS = nullptr;
    const Expr *RHS = nullptr;
    BinaryOperatorKind Operator;
  };

  struct UnaryOpInfo {
    const Expr *FoundExpr = nullptr;
    const Expr *SubExpr = nullptr;
    UnaryOperatorKind Operator;

    bool IsIncrementOp() {
      return Operator == UO_PostInc || Operator == UO_PreInc;
    }
  };

  std::optional<UnaryOpInfo> GetUnaryOperatorInfo(const Expr *E) {
    // If this is a simple unary operator, just return its details.
    if (const auto *UO = dyn_cast<UnaryOperator>(E))
      return UnaryOpInfo{UO, UO->getSubExpr()->IgnoreImpCasts(),
                         UO->getOpcode()};

    // This might be an overloaded operator or a dependent context, so make sure
    // we can get as many details out of this as we can.
    if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(E)) {
      UnaryOpInfo Inf;
      Inf.FoundExpr = OpCall;

      switch (OpCall->getOperator()) {
      default:
        return std::nullopt;
      case OO_PlusPlus:
        Inf.Operator = OpCall->getNumArgs() == 1 ? UO_PreInc : UO_PostInc;
        break;
      case OO_MinusMinus:
        Inf.Operator = OpCall->getNumArgs() == 1 ? UO_PreDec : UO_PostDec;
        break;
      case OO_Amp:
        Inf.Operator = UO_AddrOf;
        break;
      case OO_Star:
        Inf.Operator = UO_Deref;
        break;
      case OO_Plus:
        Inf.Operator = UO_Plus;
        break;
      case OO_Minus:
        Inf.Operator = UO_Minus;
        break;
      case OO_Tilde:
        Inf.Operator = UO_Not;
        break;
      case OO_Exclaim:
        Inf.Operator = UO_LNot;
        break;
      case OO_Coawait:
        Inf.Operator = UO_Coawait;
        break;
      }

      // Some of the above can be both binary and unary operations, so make sure
      // we get the right one.
      if (Inf.Operator != UO_PostInc && Inf.Operator != UO_PostDec &&
          OpCall->getNumArgs() != 1)
        return std::nullopt;

      Inf.SubExpr = OpCall->getArg(0);
      return Inf;
    }
    return std::nullopt;
  }

  // Get a normalized version of a binary operator.
  std::optional<BinaryOpInfo> GetBinaryOperatorInfo(const Expr *E) {
    if (const auto *BO = dyn_cast<BinaryOperator>(E))
      return BinaryOpInfo{BO, BO->getLHS()->IgnoreImpCasts(),
                          BO->getRHS()->IgnoreImpCasts(), BO->getOpcode()};

    // In case this is an operator-call, which allows us to support overloaded
    // operators and dependent expression.
    if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(E)) {
      BinaryOpInfo Inf;
      Inf.FoundExpr = OpCall;

      switch (OpCall->getOperator()) {
      default:
        return std::nullopt;
      case OO_Plus:
        Inf.Operator = BO_Add;
        break;
      case OO_Minus:
        Inf.Operator = BO_Sub;
        break;
      case OO_Star:
        Inf.Operator = BO_Mul;
        break;
      case OO_Slash:
        Inf.Operator = BO_Div;
        break;
      case OO_Percent:
        Inf.Operator = BO_Rem;
        break;
      case OO_Caret:
        Inf.Operator = BO_Xor;
        break;
      case OO_Amp:
        Inf.Operator = BO_And;
        break;
      case OO_Pipe:
        Inf.Operator = BO_Or;
        break;
      case OO_Equal:
        Inf.Operator = BO_Assign;
        break;
      case OO_Spaceship:
        Inf.Operator = BO_Cmp;
        break;
      case OO_Less:
        Inf.Operator = BO_LT;
        break;
      case OO_Greater:
        Inf.Operator = BO_GT;
        break;
      case OO_PlusEqual:
        Inf.Operator = BO_AddAssign;
        break;
      case OO_MinusEqual:
        Inf.Operator = BO_SubAssign;
        break;
      case OO_StarEqual:
        Inf.Operator = BO_MulAssign;
        break;
      case OO_SlashEqual:
        Inf.Operator = BO_DivAssign;
        break;
      case OO_PercentEqual:
        Inf.Operator = BO_RemAssign;
        break;
      case OO_CaretEqual:
        Inf.Operator = BO_XorAssign;
        break;
      case OO_AmpEqual:
        Inf.Operator = BO_AndAssign;
        break;
      case OO_PipeEqual:
        Inf.Operator = BO_OrAssign;
        break;
      case OO_LessLess:
        Inf.Operator = BO_Shl;
        break;
      case OO_GreaterGreater:
        Inf.Operator = BO_Shr;
        break;
      case OO_LessLessEqual:
        Inf.Operator = BO_ShlAssign;
        break;
      case OO_GreaterGreaterEqual:
        Inf.Operator = BO_ShrAssign;
        break;
      case OO_EqualEqual:
        Inf.Operator = BO_EQ;
        break;
      case OO_ExclaimEqual:
        Inf.Operator = BO_NE;
        break;
      case OO_LessEqual:
        Inf.Operator = BO_LE;
        break;
      case OO_GreaterEqual:
        Inf.Operator = BO_GE;
        break;
      case OO_AmpAmp:
        Inf.Operator = BO_LAnd;
        break;
      case OO_PipePipe:
        Inf.Operator = BO_LOr;
        break;
      case OO_Comma:
        Inf.Operator = BO_Comma;
        break;
      case OO_ArrowStar:
        Inf.Operator = BO_PtrMemI;
        break;
      }

      // This isn't a binary operator unless there are two arguments.
      if (OpCall->getNumArgs() != 2)
        return std::nullopt;

      // Callee is the call-operator, so we only need to extract the two
      // arguments here.
      Inf.LHS = OpCall->getArg(0)->IgnoreImpCasts();
      Inf.RHS = OpCall->getArg(1)->IgnoreImpCasts();
      return Inf;
    }

    return std::nullopt;
  }

  // Checks a required assignment operation, but don't check the LHS or RHS,
  // callers have to do that here.
  std::optional<BinaryOpInfo> CheckAssignment(const Expr *E) {
    std::optional<BinaryOpInfo> Inf = GetBinaryOperatorInfo(E);

    if (!Inf) {
      DiagnoseInvalidAtomic(E->getExprLoc(),
                            SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                                << diag::OACCAtomicExpr::Assign);
      return std::nullopt;
    }

    if (Inf->Operator != BO_Assign) {
      DiagnoseInvalidAtomic(Inf->FoundExpr->getExprLoc(),
                            SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                                << diag::OACCAtomicExpr::Assign);
      return std::nullopt;
    }

    // Assignment always requires an lvalue/scalar on the LHS.
    if (CheckOperandVariable(
            Inf->LHS, SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
                          << /*left=*/0 << diag::OACCAtomicOpKind::Assign))
      return std::nullopt;

    return Inf;
  }

  struct IDACInfo {
    bool Failed = false;
    enum ExprKindTy {
      Invalid,
      // increment/decrement ops.
      Unary,
      // v = x
      SimpleAssign,
      // x = expr
      ExprAssign,
      // x binop= expr
      CompoundAssign,
      // x = x binop expr
      // x = expr binop x
      AssignBinOp
    } ExprKind;

    // The variable referred to as 'x' in all of the grammar, such that it is
    // needed in compound statement checking of capture to check between the two
    // expressions.
    const Expr *X_Var = nullptr;

    static IDACInfo Fail() { return IDACInfo{true, Invalid, nullptr}; };
  };

  // Helper for CheckIncDecAssignCompoundAssign, does checks for inc/dec.
  IDACInfo CheckIncDec(UnaryOpInfo Inf) {

    if (!UnaryOperator::isIncrementDecrementOp(Inf.Operator)) {
      DiagnoseInvalidAtomic(
          Inf.FoundExpr->getExprLoc(),
          SemaRef.PDiag(diag::note_acc_atomic_unsupported_unary_operator));
      return IDACInfo::Fail();
    }
    bool Failed = CheckOperandVariable(
        Inf.SubExpr,
        SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
            << /*none=*/2
            << (Inf.IsIncrementOp() ? diag::OACCAtomicOpKind::Inc
                                    : diag::OACCAtomicOpKind::Dec));
    // For increment/decrements, the subexpr is the 'x' (x++, ++x, etc).
    return IDACInfo{Failed, IDACInfo::Unary, Inf.SubExpr};
  }

  enum class SimpleAssignKind { None, Var, Expr };

  // Check an assignment, and ensure the RHS is either x binop expr or expr
  // binop x.
  // If AllowSimpleAssign, also allows v = x;
  IDACInfo CheckAssignmentWithBinOpOnRHS(BinaryOpInfo AssignInf,
                                         SimpleAssignKind SAK) {
    PartialDiagnostic PD =
        SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
        << /*left=*/0 << diag::OACCAtomicOpKind::Assign;
    if (CheckOperandVariable(AssignInf.LHS, PD))
      return IDACInfo::Fail();

    std::optional<BinaryOpInfo> BinInf = GetBinaryOperatorInfo(AssignInf.RHS);

    if (!BinInf) {

      // Capture in a compound statement allows v = x assignment.  So make sure
      // we permit that here.
      if (SAK != SimpleAssignKind::None) {
        PartialDiagnostic PD =
            SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
            << /*right=*/1 << diag::OACCAtomicOpKind::Assign;
        if (SAK == SimpleAssignKind::Var) {
          // In the var version, everywhere we allow v = x;, X is the RHS.
          return IDACInfo{CheckOperandVariable(AssignInf.RHS, PD),
                          IDACInfo::SimpleAssign, AssignInf.RHS};
        }
        assert(SAK == SimpleAssignKind::Expr);
        // In the expression version, supported by v=x; x = expr;, we need to
        // set to the LHS here.
        return IDACInfo{CheckOperandExpr(AssignInf.RHS, PD),
                        IDACInfo::ExprAssign, AssignInf.LHS};
      }

      DiagnoseInvalidAtomic(
          AssignInf.RHS->getExprLoc(),
          SemaRef.PDiag(diag::note_acc_atomic_expected_binop));

      return IDACInfo::Fail();
    }
    switch (BinInf->Operator) {
    default:
      DiagnoseInvalidAtomic(
          BinInf->FoundExpr->getExprLoc(),
          SemaRef.PDiag(diag::note_acc_atomic_unsupported_binary_operator));
      return IDACInfo::Fail();
      // binop is one of +, *, -, /, &, ^, |, <<, or >>
    case BO_Add:
    case BO_Mul:
    case BO_Sub:
    case BO_Div:
    case BO_And:
    case BO_Xor:
    case BO_Or:
    case BO_Shl:
    case BO_Shr:
      // Handle these outside of the switch.
      break;
    }

    llvm::FoldingSetNodeID LHS_ID, InnerLHS_ID, InnerRHS_ID;
    AssignInf.LHS->Profile(LHS_ID, SemaRef.getASTContext(),
                           /*Canonical=*/true);
    BinInf->LHS->Profile(InnerLHS_ID, SemaRef.getASTContext(),
                         /*Canonical=*/true);

    // This is X = X binop expr;
    // Check the RHS is an expression.
    if (LHS_ID == InnerLHS_ID)
      return IDACInfo{
          CheckOperandExpr(
              BinInf->RHS,
              SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar
                            << /*right=*/1
                            << diag::OACCAtomicOpKind::CompoundAssign)),
          IDACInfo::AssignBinOp, AssignInf.LHS};

    BinInf->RHS->Profile(InnerRHS_ID, SemaRef.getASTContext(),
                         /*Canonical=*/true);
    // This is X = expr binop X;
    // Check the LHS is an expression
    if (LHS_ID == InnerRHS_ID)
      return IDACInfo{
          CheckOperandExpr(
              BinInf->LHS,
              SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
                  << /*left=*/0 << diag::OACCAtomicOpKind::CompoundAssign),
          IDACInfo::AssignBinOp, AssignInf.LHS};

    // If nothing matches, error out.
    DiagnoseInvalidAtomic(BinInf->FoundExpr->getExprLoc(),
                          SemaRef.PDiag(diag::note_acc_atomic_mismatch_operand)
                              << const_cast<Expr *>(AssignInf.LHS)
                              << const_cast<Expr *>(BinInf->LHS)
                              << const_cast<Expr *>(BinInf->RHS));
    return IDACInfo::Fail();
  }

  // Ensures that the expression is an increment/decrement, an assignment, or a
  // compound assignment. If its an assignment, allows the x binop expr/x binop
  // expr syntax. If it is a compound-assignment, allows any expr on the RHS.
  IDACInfo CheckIncDecAssignCompoundAssign(const Expr *E,
                                           SimpleAssignKind SAK) {
    std::optional<UnaryOpInfo> UInf = GetUnaryOperatorInfo(E);

    // If this is a unary operator, only increment/decrement are allowed, so get
    // unary operator, then check everything we can.
    if (UInf)
      return CheckIncDec(*UInf);

    std::optional<BinaryOpInfo> BinInf = GetBinaryOperatorInfo(E);

    // Unary or binary operator were the only choices, so error here.
    if (!BinInf) {
      DiagnoseInvalidAtomic(E->getExprLoc(),
                            SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                                << diag::OACCAtomicExpr::UnaryCompAssign);
      return IDACInfo::Fail();
    }

    switch (BinInf->Operator) {
    default:
      DiagnoseInvalidAtomic(
          BinInf->FoundExpr->getExprLoc(),
          SemaRef.PDiag(
              diag::note_acc_atomic_unsupported_compound_binary_operator));
      return IDACInfo::Fail();
    case BO_Assign:
      return CheckAssignmentWithBinOpOnRHS(*BinInf, SAK);
    case BO_AddAssign:
    case BO_MulAssign:
    case BO_SubAssign:
    case BO_DivAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
    case BO_ShlAssign:
    case BO_ShrAssign: {
      PartialDiagnostic LPD =
          SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
          << /*left=*/0 << diag::OACCAtomicOpKind::CompoundAssign;
      PartialDiagnostic RPD =
          SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
          << /*right=*/1 << diag::OACCAtomicOpKind::CompoundAssign;
      // nothing to do other than check the variable expressions.
      // success or failure
      bool Failed = CheckOperandVariable(BinInf->LHS, LPD) ||
                    CheckOperandExpr(BinInf->RHS, RPD);

      return IDACInfo{Failed, IDACInfo::CompoundAssign, BinInf->LHS};
    }
    }
    llvm_unreachable("all binary operator kinds should be checked above");
  }

  StmtResult CheckRead() {
    Expr *AssocExpr = RequireExpr(
        AssocStmt.get(), SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                             << diag::OACCAtomicExpr::Assign);

    if (!AssocExpr)
      return getRecoveryExpr();

    std::optional<BinaryOpInfo> AssignRes = CheckAssignment(AssocExpr);
    if (!AssignRes)
      return getRecoveryExpr();

    PartialDiagnostic PD =
        SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
        << /*right=*/1 << diag::OACCAtomicOpKind::Assign;

    // Finally, check the RHS.
    if (CheckOperandVariable(AssignRes->RHS, PD))
      return getRecoveryExpr();

    return AssocStmt;
  }

  StmtResult CheckWrite() {
    Expr *AssocExpr = RequireExpr(
        AssocStmt.get(), SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                             << diag::OACCAtomicExpr::Assign);

    if (!AssocExpr)
      return getRecoveryExpr();

    std::optional<BinaryOpInfo> AssignRes = CheckAssignment(AssocExpr);
    if (!AssignRes)
      return getRecoveryExpr();

    PartialDiagnostic PD =
        SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
        << /*right=*/1 << diag::OACCAtomicOpKind::Assign;

    // Finally, check the RHS.
    if (CheckOperandExpr(AssignRes->RHS, PD))
      return getRecoveryExpr();

    return AssocStmt;
  }

  StmtResult CheckUpdate() {
    Expr *AssocExpr = RequireExpr(
        AssocStmt.get(), SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                             << diag::OACCAtomicExpr::UnaryCompAssign);

    if (!AssocExpr ||
        CheckIncDecAssignCompoundAssign(AssocExpr, SimpleAssignKind::None)
            .Failed)
      return getRecoveryExpr();

    return AssocStmt;
  }

  bool CheckVarRefsSame(IDACInfo::ExprKindTy FirstKind, const Expr *FirstX,
                        IDACInfo::ExprKindTy SecondKind, const Expr *SecondX) {
    llvm::FoldingSetNodeID First_ID, Second_ID;
    FirstX->Profile(First_ID, SemaRef.getASTContext(), /*Canonical=*/true);
    SecondX->Profile(Second_ID, SemaRef.getASTContext(), /*Canonical=*/true);

    if (First_ID == Second_ID)
      return false;

    PartialDiagnostic PD =
        SemaRef.PDiag(diag::note_acc_atomic_mismatch_compound_operand)
        << FirstKind << const_cast<Expr *>(FirstX) << SecondKind
        << const_cast<Expr *>(SecondX);

    return DiagnoseInvalidAtomic(SecondX->getExprLoc(), PD);
  }

  StmtResult CheckCapture() {
    if (const auto *CmpdStmt = dyn_cast<CompoundStmt>(AssocStmt.get())) {
      auto *const *BodyItr = CmpdStmt->body().begin();
      PartialDiagnostic PD = SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                             << diag::OACCAtomicExpr::UnaryCompAssign;
      // If we don't have at least 1 statement, error.
      if (BodyItr == CmpdStmt->body().end()) {
        DiagnoseInvalidAtomic(CmpdStmt->getBeginLoc(), PD);
        return getRecoveryExpr();
      }

      // First Expr can be inc/dec, assign, or compound assign.
      Expr *FirstExpr = RequireExpr(*BodyItr, PD);
      if (!FirstExpr)
        return getRecoveryExpr();

      IDACInfo FirstExprResults =
          CheckIncDecAssignCompoundAssign(FirstExpr, SimpleAssignKind::Var);
      if (FirstExprResults.Failed)
        return getRecoveryExpr();

      ++BodyItr;

      // If we don't have second statement, error.
      if (BodyItr == CmpdStmt->body().end()) {
        DiagnoseInvalidAtomic(CmpdStmt->getEndLoc(), PD);
        return getRecoveryExpr();
      }

      Expr *SecondExpr = RequireExpr(*BodyItr, PD);
      if (!SecondExpr)
        return getRecoveryExpr();

      assert(FirstExprResults.ExprKind != IDACInfo::Invalid);

      switch (FirstExprResults.ExprKind) {
      case IDACInfo::Invalid:
      case IDACInfo::ExprAssign:
        llvm_unreachable("Should have error'ed out by now");
      case IDACInfo::Unary:
      case IDACInfo::CompoundAssign:
      case IDACInfo::AssignBinOp: {
        // Everything but simple-assign can only be followed by a simple
        // assignment.
        std::optional<BinaryOpInfo> AssignRes = CheckAssignment(SecondExpr);
        if (!AssignRes)
          return getRecoveryExpr();

        PartialDiagnostic PD =
            SemaRef.PDiag(diag::note_acc_atomic_operand_lvalue_scalar)
            << /*right=*/1 << diag::OACCAtomicOpKind::Assign;

        if (CheckOperandVariable(AssignRes->RHS, PD))
          return getRecoveryExpr();

        if (CheckVarRefsSame(FirstExprResults.ExprKind, FirstExprResults.X_Var,
                             IDACInfo::SimpleAssign, AssignRes->RHS))
          return getRecoveryExpr();
        break;
      }
      case IDACInfo::SimpleAssign: {
        // If the first was v = x, anything but simple expression is allowed.
        IDACInfo SecondExprResults =
            CheckIncDecAssignCompoundAssign(SecondExpr, SimpleAssignKind::Expr);
        if (SecondExprResults.Failed)
          return getRecoveryExpr();

        if (CheckVarRefsSame(FirstExprResults.ExprKind, FirstExprResults.X_Var,
                             SecondExprResults.ExprKind,
                             SecondExprResults.X_Var))
          return getRecoveryExpr();
        break;
      }
      }
      ++BodyItr;
      if (BodyItr != CmpdStmt->body().end()) {
        DiagnoseInvalidAtomic(
            (*BodyItr)->getBeginLoc(),
            SemaRef.PDiag(diag::note_acc_atomic_too_many_stmts));
        return getRecoveryExpr();
      }
    } else {
      // This check doesn't need to happen if it is a compound stmt.
      Expr *AssocExpr = RequireExpr(
          AssocStmt.get(), SemaRef.PDiag(diag::note_acc_atomic_expr_must_be)
                               << diag::OACCAtomicExpr::Assign);
      if (!AssocExpr)
        return getRecoveryExpr();

      // First, we require an assignment.
      std::optional<BinaryOpInfo> AssignRes = CheckAssignment(AssocExpr);

      if (!AssignRes)
        return getRecoveryExpr();

      if (CheckIncDecAssignCompoundAssign(AssignRes->RHS,
                                          SimpleAssignKind::None)
              .Failed)
        return getRecoveryExpr();
    }

    return AssocStmt;
  }

public:
  AtomicOperandChecker(SemaOpenACC &S, OpenACCAtomicKind AtKind,
                       SourceLocation DirLoc, StmtResult AssocStmt)
      : SemaRef(S), AtKind(AtKind), AtomicDirLoc(DirLoc), AssocStmt(AssocStmt) {
  }

  StmtResult Check() {

    switch (AtKind) {
    case OpenACCAtomicKind::Read:
      return CheckRead();
    case OpenACCAtomicKind::Write:
      return CheckWrite();
    case OpenACCAtomicKind::None:
    case OpenACCAtomicKind::Update:
      return CheckUpdate();
    case OpenACCAtomicKind::Capture:
      return CheckCapture();
    }
    llvm_unreachable("Unhandled atomic kind?");
  }
};
} // namespace

StmtResult SemaOpenACC::CheckAtomicAssociatedStmt(SourceLocation AtomicDirLoc,
                                                  OpenACCAtomicKind AtKind,
                                                  StmtResult AssocStmt) {
  if (!AssocStmt.isUsable())
    return AssocStmt;

  if (isa<RecoveryExpr>(AssocStmt.get()))
    return AssocStmt;

  AtomicOperandChecker Checker{*this, AtKind, AtomicDirLoc, AssocStmt};
  return Checker.Check();
}
