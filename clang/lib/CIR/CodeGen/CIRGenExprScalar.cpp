//===--- CIRGenExprScalar.cpp - Emit CIR Code for Scalar Exprs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with scalar CIR types as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {

class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
  CIRGenFunction &CGF;
  mlir::OpBuilder &Builder;
  bool IgnoreResultAssign;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, mlir::OpBuilder &builder,
                    bool ira = false)
      : CGF(cgf), Builder(builder), IgnoreResultAssign(ira) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool TestAndClearIgnoreResultAssign() {
    bool I = IgnoreResultAssign;
    IgnoreResultAssign = false;
    return I;
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *E) {
    return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
  }

  mlir::Value VisitStmt(Stmt *S) {
    S->dump(llvm::errs(), CGF.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  mlir::Value VisitExpr(Expr *E) {
    // Crashing here for "ScalarExprClassName"? Please implement
    // VisitScalarExprClassName(...) to get this working.
    emitError(CGF.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
        << E->getStmtClassName() << "'";
    assert(0 && "shouldn't be here!");
    return {};
  }

  mlir::Value VisitConstantExpr(ConstantExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitParenExpr(ParenExpr *PE) { llvm_unreachable("NYI"); }
  mlir::Value
  VisitSubstnonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *S) { llvm_unreachable("NYI"); }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *S) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }

  // Leaves.
  mlir::Value VisitIntegerLiteral(const IntegerLiteral *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        Builder.getIntegerAttr(Ty, E->getValue()));
  }

  mlir::Value VisitFixedPointLiteral(const FixedPointLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitFloatingLiteral(const FloatingLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCharacterLiteral(const CharacterLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty, Builder.getBoolAttr(E->getValue()));
  }

  mlir::Value VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitGNUNullExpr(const GNUNullExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOffsetOfExpr(OffsetOfExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitAddrLabelExpr(const AddrLabelExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSizeOfPackExpr(SizeOfPackExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSYCLUniqueStableNameExpr(SYCLUniqueStableNameExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    llvm_unreachable("NYI");
  }

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value buildLoadOfLValue(const Expr *E) {
    LValue LV = CGF.buildLValue(E);
    auto load = Builder.create<mlir::cir::LoadOp>(CGF.getLoc(E->getExprLoc()),
                                                  CGF.getCIRType(E->getType()),
                                                  LV.getPointer());
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return load;
  }

  mlir::Value buildLoadOfLValue(LValue LV, SourceLocation Loc) {
    return CGF.buildLoadOfLValue(LV, Loc).getScalarVal();
  }

  // l-values
  mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
    // FIXME: we could try to emit this as constant first, see
    // CGF.tryEmitAsConstant(E)
    return buildLoadOfLValue(E);
  }

  mlir::Value VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCIVarRefExpr(ObjCIvarRefExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCIsaExpr(ObjCIsaExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    // Do we need anything like TestAndClearIgnoreResultAssign()?
    assert(!E->getBase()->getType()->isVectorType() &&
           "vector types not implemented");

    // Emit subscript expressions in rvalue context's.  For most cases, this
    // just loads the lvalue formed by the subscript expr.  However, we have to
    // be careful, because the base of a vector subscript is occasionally an
    // rvalue, so we can't get it as an lvalue.
    return buildLoadOfLValue(E);
  }

  mlir::Value VisitMatrixSubscriptExpr(MatrixSubscriptExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitConvertVectorExpr(ConvertVectorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitMemberExpr(MemberExpr *E);
  mlir::Value VisitExtVectorelementExpr(Expr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitCompoundLiteralEpxr(CompoundLiteralExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitInitListExpr(InitListExpr *E);

  mlir::Value VisitArrayInitIndexExpr(ArrayInitIndexExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitExplicitCastExpr(ExplicitCastExpr *E) {
    return VisitCastExpr(E);
  }
  mlir::Value VisitCastExpr(CastExpr *E);
  mlir::Value VisitCallExpr(const CallExpr *E);
  mlir::Value VisitStmtExpr(StmtExpr *E) { llvm_unreachable("NYI"); }

  // Unary Operators.
  mlir::Value VisitUnaryPostDec(const UnaryOperator *E) {
    return buildScalarPrePostIncDec(E);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *E) {
    return buildScalarPrePostIncDec(E);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *E) {
    return buildScalarPrePostIncDec(E);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *E) {
    return buildScalarPrePostIncDec(E);
  }
  mlir::Value buildScalarPrePostIncDec(const UnaryOperator *E) {
    QualType type = E->getSubExpr()->getType();

    auto LV = CGF.buildLValue(E->getSubExpr());
    mlir::Value Value;
    mlir::Value Input;

    if (const AtomicType *atomicTy = type->getAs<AtomicType>()) {
      assert(0 && "no atomics inc/dec yet");
    } else {
      Value = buildLoadOfLValue(LV, E->getExprLoc());
      Input = Value;
    }

    // NOTE: When possible, more frequent cases are handled first.

    // Special case of integer increment that we have to check first: bool++.
    // Due to promotion rules, we get:
    //   bool++ -> bool = bool + 1
    //          -> bool = (int)bool + 1
    //          -> bool = ((int)bool + 1 != 0)
    // An interesting aspect of this is that increment is always true.
    // Decrement does not have this property.
    if (E->isIncrementOp() && type->isBooleanType()) {
      assert(0 && "inc simplification for booleans not implemented yet");

      // NOTE: We likely want the code below, but loading/store booleans need to
      // work first. See CIRGenFunction::buildFromMemory().
      Value = Builder.create<mlir::cir::ConstantOp>(CGF.getLoc(E->getExprLoc()),
                                                    CGF.getCIRType(type),
                                                    Builder.getBoolAttr(true));
    } else if (type->isIntegerType()) {
      bool canPerformLossyDemotionCheck = false;
      if (CGF.getContext().isPromotableIntegerType(type)) {
        canPerformLossyDemotionCheck = true;
        assert(0 && "no promotable integer inc/dec yet");
      }

      if (CGF.SanOpts.hasOneOf(
              SanitizerKind::ImplicitIntegerArithmeticValueChange) &&
          canPerformLossyDemotionCheck) {
        assert(0 &&
               "perform lossy demotion case for inc/dec not implemented yet");
      } else if (E->canOverflow() && type->isSignedIntegerOrEnumerationType()) {
        Value = buildIncDecConsiderOverflowBehavior(E, Value);
      } else if (E->canOverflow() && type->isUnsignedIntegerType() &&
                 CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) {
        assert(0 &&
               "unsigned integer overflow sanitized inc/dec not implemented");
      } else {
        auto Kind = E->isIncrementOp() ? mlir::cir::UnaryOpKind::Inc
                                       : mlir::cir::UnaryOpKind::Dec;
        Value = buildUnaryOp(E, Kind, Input);
      }
    } else if (const PointerType *ptr = type->getAs<PointerType>()) {
      assert(0 && "no pointer inc/dec yet");
    } else if (type->isVectorType()) {
      assert(0 && "no vector inc/dec yet");
    } else if (type->isRealFloatingType()) {
      assert(0 && "no float inc/dec yet");
    } else if (type->isFixedPointType()) {
      assert(0 && "no fixed point inc/dec yet");
    } else {
      assert(type->castAs<ObjCObjectPointerType>());
      assert(0 && "no objc pointer type inc/dec yet");
    }

    CIRGenFunction::SourceLocRAIIObject sourceloc{
        CGF, CGF.getLoc(E->getSourceRange())};

    if (LV.isBitField())
      assert(0 && "no bitfield inc/dec yet");
    else
      CGF.buildStoreThroughLValue(RValue::get(Value), LV);

    return E->isPrefix() ? Value : Input;
  }

  mlir::Value buildIncDecConsiderOverflowBehavior(const UnaryOperator *E,
                                                  mlir::Value V) {
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined: {
      auto Kind = E->isIncrementOp() ? mlir::cir::UnaryOpKind::Inc
                                     : mlir::cir::UnaryOpKind::Dec;
      return buildUnaryOp(E, Kind, V);
      break;
    }
    case LangOptions::SOB_Undefined:
      assert(0 &&
             "inc/dec overflow behavior SOB_Undefined not implemented yet");
      break;
    case LangOptions::SOB_Trapping:
      assert(0 && "inc/dec overflow behavior SOB_Trapping not implemented yet");
      break;
    }
    llvm_unreachable("Unknown SignedOverflowBehaviorTy");
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *E) {
    assert(!llvm::isa<MemberPointerType>(E->getType()) && "not implemented");
    return CGF.buildLValue(E->getSubExpr()).getPointer();
  }

  mlir::Value VisitUnaryDeref(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryPlus(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryMinus(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryNot(const UnaryOperator *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryLNot(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryReal(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryImag(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryExtension(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value buildUnaryOp(const UnaryOperator *E, mlir::cir::UnaryOpKind kind,
                           mlir::Value input) {
    return Builder.create<mlir::cir::UnaryOp>(
        CGF.getLoc(E->getSourceRange().getBegin()),
        CGF.getCIRType(E->getType()), kind, input);
  }

  // C++
  mlir::Value VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSourceLocExpr(SourceLocExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    CIRGenFunction::CXXDefaultInitExprScope Scope(CGF, DIE);
    return Visit(DIE->getExpr());
  }

  mlir::Value VisitCXXThisExpr(CXXThisExpr *TE) {
    auto *t = CGF.LoadCXXThis();
    assert(t->getNumResults() == 1);
    return t->getOpResult(0);
  }

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXNewExpr(const CXXNewExpr *E) {
    return CGF.buildCXXNewExpr(E);
  }
  mlir::Value VisitCXXDeleteExpr(const CXXDeleteExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitTypeTraitExpr(const TypeTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value
  VisitConceptSpecializationExpr(const ConceptSpecializationExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitRequiresExpr(const RequiresExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXThrowExpr(CXXThrowExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
    llvm_unreachable("NYI");
  }

  // Comparisons.
#define VISITCOMP(CODE)                                                        \
  mlir::Value VisitBin##CODE(const BinaryOperator *E) { return buildCmp(E); }
  VISITCOMP(LT)
  VISITCOMP(GT)
  VISITCOMP(LE)
  VISITCOMP(GE)
  VISITCOMP(EQ)
  VISITCOMP(NE)
#undef VISITCOMP

  mlir::Value VisitBinAssign(const BinaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitBinLAnd(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitBinLOr(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitBinComma(const BinaryOperator *E) {
    CGF.buildIgnoredExpr(E->getLHS());
    // NOTE: We don't need to EnsureInsertPoint() like LLVM codegen.
    return Visit(E->getRHS());
  }

  mlir::Value VisitBinPtrMemD(const Expr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitBinPtrMemI(const Expr *E) { llvm_unreachable("NYI"); }

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *E) {
    llvm_unreachable("NYI");
  }

  // Other Operators.
  mlir::Value VisitBlockExpr(const BlockExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitChooseExpr(ChooseExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitVAArgExpr(VAArgExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCStringLiteral(const ObjCStringLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCBoxedExpr(ObjCBoxedExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitAsTypeExpr(AsTypeExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitAtomicExpr(AtomicExpr *E) { llvm_unreachable("NYI"); }

  // Emit a conversion from the specified type to the specified destination
  // type, both of which are CIR scalar types.
  struct ScalarConversionOpts {
    bool TreatBooleanAsSigned;
    bool EmitImplicitIntegerTruncationChecks;
    bool EmitImplicitIntegerSignChangeChecks;

    ScalarConversionOpts()
        : TreatBooleanAsSigned(false),
          EmitImplicitIntegerTruncationChecks(false),
          EmitImplicitIntegerSignChangeChecks(false) {}

    ScalarConversionOpts(clang::SanitizerSet SanOpts)
        : TreatBooleanAsSigned(false),
          EmitImplicitIntegerTruncationChecks(
              SanOpts.hasOneOf(SanitizerKind::ImplicitIntegerTruncation)),
          EmitImplicitIntegerSignChangeChecks(
              SanOpts.has(SanitizerKind::ImplicitIntegerSignChange)) {}
  };
  mlir::Value buildScalarCast(mlir::Value Src, QualType SrcType,
                              QualType DstType, mlir::Type SrcTy,
                              mlir::Type DstTy, ScalarConversionOpts Opts);

  struct BinOpInfo {
    mlir::Value LHS;
    mlir::Value RHS;
    SourceRange Loc;
    QualType Ty;                   // Computation Type.
    BinaryOperator::Opcode Opcode; // Opcode of BinOp to perform
    FPOptions FPFeatures;
    const Expr *E; // Entire expr, for error unsupported.  May not be binop.

    /// Check if the binop computes a division or a remainder.
    bool isDivremOp() const {
      return Opcode == BO_Div || Opcode == BO_Rem || Opcode == BO_DivAssign ||
             Opcode == BO_RemAssign;
    }

    /// Check if at least one operand is a fixed point type. In such cases,
    /// this operation did not follow usual arithmetic conversion and both
    /// operands might not be of the same type.
    bool isFixedPointOp() const {
      // We cannot simply check the result type since comparison operations
      // return an int.
      if (const auto *BinOp = llvm::dyn_cast<BinaryOperator>(E)) {
        QualType LHSType = BinOp->getLHS()->getType();
        QualType RHSType = BinOp->getRHS()->getType();
        return LHSType->isFixedPointType() || RHSType->isFixedPointType();
      }
      if (const auto *UnOp = llvm::dyn_cast<UnaryOperator>(E))
        return UnOp->getSubExpr()->getType()->isFixedPointType();
      return false;
    }
  };

  BinOpInfo buildBinOps(const BinaryOperator *E) {
    BinOpInfo Result;
    Result.LHS = Visit(E->getLHS());
    Result.RHS = Visit(E->getRHS());
    Result.Ty = E->getType();
    Result.Opcode = E->getOpcode();
    Result.Loc = E->getSourceRange();
    // TODO: Result.FPFeatures
    Result.E = E;
    return Result;
  }

  mlir::Value buildMul(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Mul,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildDiv(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Div,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildRem(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Rem,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildAdd(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Add,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildSub(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Sub,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildShl(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Shl,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildShr(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Shr,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildAnd(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::And,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildXor(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Xor,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildOr(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Or,
        Ops.LHS, Ops.RHS);
  }

  LValue buildCompoundAssignLValue(
      const CompoundAssignOperator *E,
      mlir::Value (ScalarExprEmitter::*F)(const BinOpInfo &),
      mlir::Value &Result);
  mlir::Value
  buildCompoundAssign(const CompoundAssignOperator *E,
                      mlir::Value (ScalarExprEmitter::*F)(const BinOpInfo &));

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    return build##OP(buildBinOps(E));                                          \
  }                                                                            \
  mlir::Value VisitBin##OP##Assign(const CompoundAssignOperator *E) {          \
    return buildCompoundAssign(E, &ScalarExprEmitter::build##OP);              \
  }

  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
  HANDLEBINOP(Rem)
  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
  HANDLEBINOP(Shl)
  HANDLEBINOP(Shr)
  HANDLEBINOP(And)
  HANDLEBINOP(Xor)
  HANDLEBINOP(Or)
#undef HANDLEBINOP

  mlir::Value buildCmp(const BinaryOperator *E) {
    mlir::Value Result;
    QualType LHSTy = E->getLHS()->getType();
    QualType RHSTy = E->getRHS()->getType();

    if (const MemberPointerType *MPT = LHSTy->getAs<MemberPointerType>()) {
      assert(0 && "not implemented");
    } else if (!LHSTy->isAnyComplexType() && !RHSTy->isAnyComplexType()) {
      BinOpInfo BOInfo = buildBinOps(E);
      mlir::Value LHS = BOInfo.LHS;
      mlir::Value RHS = BOInfo.RHS;

      if (LHSTy->isVectorType()) {
        // Cannot handle any vector just yet.
        assert(0 && "not implemented");
        // If AltiVec, the comparison results in a numeric type, so we use
        // intrinsics comparing vectors and giving 0 or 1 as a result
        if (!E->getType()->isVectorType())
          assert(0 && "not implemented");
      }
      if (BOInfo.isFixedPointOp()) {
        assert(0 && "not implemented");
      } else {
        // TODO: when we add proper basic types to CIR we
        // probably won't need to handle
        // LHSTy->hasSignedIntegerRepresentation()

        // Unsigned integers and pointers.
        if (LHS.getType().isa<mlir::cir::PointerType>() ||
            RHS.getType().isa<mlir::cir::PointerType>()) {
          // TODO: Handle StrictVTablePointers and
          // mayBeDynamicClass/invariant group.
          assert(0 && "not implemented");
        }

        mlir::cir::CmpOpKind Kind;
        switch (E->getOpcode()) {
        case BO_LT:
          Kind = mlir::cir::CmpOpKind::lt;
          break;
        case BO_GT:
          Kind = mlir::cir::CmpOpKind::gt;
          break;
        case BO_LE:
          Kind = mlir::cir::CmpOpKind::le;
          break;
        case BO_GE:
          Kind = mlir::cir::CmpOpKind::ge;
          break;
        case BO_EQ:
          Kind = mlir::cir::CmpOpKind::eq;
          break;
        case BO_NE:
          Kind = mlir::cir::CmpOpKind::ne;
          break;
        default:
          llvm_unreachable("unsupported");
        }

        return Builder.create<mlir::cir::CmpOp>(CGF.getLoc(BOInfo.Loc),
                                                CGF.getCIRType(BOInfo.Ty), Kind,
                                                BOInfo.LHS, BOInfo.RHS);
      }

      // If this is a vector comparison, sign extend the result to the
      // appropriate vector integer type and return it (don't convert to
      // bool).
      if (LHSTy->isVectorType())
        assert(0 && "not implemented");
    } else { // Complex Comparison: can only be an equality comparison.
      assert(0 && "not implemented");
    }

    return buildScalarConversion(Result, CGF.getContext().BoolTy, E->getType(),
                                 E->getExprLoc());
  }

  mlir::Value buildIntToBoolConversion(mlir::Value srcVal, mlir::Location loc) {
    // Because of the type rules of C, we often end up computing a
    // logical value, then zero extending it to int, then wanting it
    // as a logical value again.
    // TODO: optimize this common case here or leave it for later
    // CIR passes?
    mlir::Type boolTy = CGF.getCIRType(CGF.getContext().BoolTy);
    return Builder.create<mlir::cir::CastOp>(
        loc, boolTy, mlir::cir::CastKind::int_to_bool, srcVal);
  }

  /// EmitConversionToBool - Convert the specified expression value to a
  /// boolean (i1) truth value.  This is equivalent to "Val != 0".
  mlir::Value buildConversionToBool(mlir::Value Src, QualType SrcType,
                                    mlir::Location loc) {
    assert(SrcType.isCanonical() && "EmitScalarConversion strips typedefs");

    if (SrcType->isRealFloatingType())
      assert(0 && "not implemented");

    if (auto *MPT = llvm::dyn_cast<MemberPointerType>(SrcType))
      assert(0 && "not implemented");

    assert((SrcType->isIntegerType() ||
            Src.getType().isa<::mlir::cir::PointerType>()) &&
           "Unknown scalar type to convert");

    assert(Src.getType().isa<mlir::IntegerType>() &&
           "pointer source not implemented");
    return buildIntToBoolConversion(Src, loc);
  }

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  /// TODO: do we need ScalarConversionOpts here? Should be done in another
  /// pass.
  mlir::Value
  buildScalarConversion(mlir::Value Src, QualType SrcType, QualType DstType,
                        SourceLocation Loc,
                        ScalarConversionOpts Opts = ScalarConversionOpts()) {
    if (SrcType->isFixedPointType()) {
      assert(0 && "not implemented");
    } else if (DstType->isFixedPointType()) {
      assert(0 && "not implemented");
    }

    SrcType = CGF.getContext().getCanonicalType(SrcType);
    DstType = CGF.getContext().getCanonicalType(DstType);
    if (SrcType == DstType)
      return Src;

    if (DstType->isVoidType())
      return nullptr;
    mlir::Type SrcTy = Src.getType();

    // Handle conversions to bool first, they are special: comparisons against
    // 0.
    if (DstType->isBooleanType())
      return buildConversionToBool(Src, SrcType, CGF.getLoc(Loc));

    mlir::Type DstTy = CGF.getCIRType(DstType);

    // Cast from half through float if half isn't a native type.
    if (SrcType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      assert(0 && "not implemented");
    }

    // TODO(cir): LLVM codegen ignore conversions like int -> uint,
    // is there anything to be done for CIR here?
    if (SrcTy == DstTy) {
      if (Opts.EmitImplicitIntegerSignChangeChecks)
        assert(0 && "not implemented");
      return Src;
    }

    // Handle pointer conversions next: pointers can only be converted to/from
    // other pointers and integers.
    if (DstTy.isa<::mlir::cir::PointerType>()) {
      assert(0 && "not implemented");
    }

    if (SrcTy.isa<::mlir::cir::PointerType>()) {
      // Must be a ptr to int cast.
      assert(DstTy.isa<mlir::IntegerType>() && "not ptr->int?");
      assert(0 && "not implemented");
    }

    // A scalar can be splatted to an extended vector of the same element type
    if (DstType->isExtVectorType() && !SrcType->isVectorType()) {
      // Sema should add casts to make sure that the source expression's type
      // is the same as the vector's element type (sans qualifiers)
      assert(DstType->castAs<ExtVectorType>()->getElementType().getTypePtr() ==
                 SrcType.getTypePtr() &&
             "Splatted expr doesn't match with vector element type?");

      assert(0 && "not implemented");
    }

    if (SrcType->isMatrixType() && DstType->isMatrixType())
      assert(0 && "not implemented");

    // Finally, we have the arithmetic types: real int/float.
    mlir::Value Res = nullptr;
    mlir::Type ResTy = DstTy;

    // An overflowing conversion has undefined behavior if eitehr the source
    // type or the destination type is a floating-point type. However, we
    // consider the range of representable values for all floating-point types
    // to be [-inf,+inf], so no overflow can ever happen when the destination
    // type is a floating-point type.
    if (CGF.SanOpts.has(SanitizerKind::FloatCastOverflow))
      llvm_unreachable("NYI");

    // Cast to half through float if half isn't a native type.
    if (DstType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      llvm_unreachable("NYI");
    }

    Res = buildScalarCast(Src, SrcType, DstType, SrcTy, DstTy, Opts);

    if (DstTy != ResTy) {
      llvm_unreachable("NYI");
    }

    if (Opts.EmitImplicitIntegerTruncationChecks)
      llvm_unreachable("NYI");

    if (Opts.EmitImplicitIntegerSignChangeChecks)
      llvm_unreachable("NYI");

    return Res;
  }
};

} // namespace

/// Emit the computation of the specified expression of scalar type,
/// ignoring the result.
mlir::Value CIRGenFunction::buildScalarExpr(const Expr *E) {
  assert(E && hasScalarEvaluationKind(E->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(E));
}

// Emit code for an explicit or implicit cast.  Implicit
// casts have to handle a more broad range of conversions than explicit
// casts, as they handle things like function to ptr-to-function decay
// etc.
mlir::Value ScalarExprEmitter::VisitCastExpr(CastExpr *CE) {
  Expr *E = CE->getSubExpr();
  QualType DestTy = CE->getType();
  CastKind Kind = CE->getCastKind();

  // These cases are generally not written to ignore the result of evaluating
  // their sub-expressions, so we clear this now.
  bool Ignored = TestAndClearIgnoreResultAssign();
  (void)Ignored;

  // Since almost all cast kinds apply to scalars, this switch doesn't have a
  // default case, so the compiler will warn on a missing case. The cases are
  // in the same order as in the CastKind enum.
  switch (Kind) {
  case clang::CK_Dependent:
    llvm_unreachable("dependent cast kind in CIR gen!");
  case clang::CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  case CK_LValueBitCast:
    llvm_unreachable("NYI");
  case CK_ObjCObjectLValueCast:
    llvm_unreachable("NYI");
  case CK_LValueToRValueBitCast:
    llvm_unreachable("NYI");
  case CK_CPointerToObjCPointerCast:
    llvm_unreachable("NYI");
  case CK_BlockPointerToObjCPointerCast:
    llvm_unreachable("NYI");
  case CK_AnyPointerToBlockPointerCast:
    llvm_unreachable("NYI");
  case CK_BitCast:
    llvm_unreachable("NYI");
  case CK_AddressSpaceConversion:
    llvm_unreachable("NYI");
  case CK_AtomicToNonAtomic:
    llvm_unreachable("NYI");
  case CK_NonAtomicToAtomic:
    llvm_unreachable("NYI");
  case CK_UserDefinedConversion:
    llvm_unreachable("NYI");
  case CK_NoOp: {
    auto V = Visit(const_cast<Expr *>(E));
    if (V) {
      // CK_NoOp can model a pointer qualification conversion, which can remove
      // an array bound and change the IR type.
      // FIXME: Once pointee types are removed from IR, remove this.
      auto T = CGF.convertType(DestTy);
      if (T != V.getType())
        assert(0 && "NYI");
    }
    return V;
  }
  case CK_BaseToDerived:
    llvm_unreachable("NYI");
  case CK_DerivedToBase:
    llvm_unreachable("NYI");
  case CK_Dynamic:
    llvm_unreachable("NYI");
  case CK_ArrayToPointerDecay:
    return CGF.buildArrayToPointerDecay(E).getPointer();
  case CK_FunctionToPointerDecay:
    llvm_unreachable("NYI");

  case CK_NullToPointer: {
    // FIXME: use MustVisitNullValue(E) and evaluate expr.
    // Note that DestTy is used as the MLIR type instead of a custom
    // nullptr type.
    mlir::Type Ty = CGF.getCIRType(DestTy);
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        mlir::cir::NullAttr::get(Builder.getContext(), Ty));
  }

  case CK_NullToMemberPointer:
    llvm_unreachable("NYI");
  case CK_ReinterpretMemberPointer:
    llvm_unreachable("NYI");
  case CK_BaseToDerivedMemberPointer:
    llvm_unreachable("NYI");
  case CK_DerivedToBaseMemberPointer:
    llvm_unreachable("NYI");
  case CK_ARCProduceObject:
    llvm_unreachable("NYI");
  case CK_ARCConsumeObject:
    llvm_unreachable("NYI");
  case CK_ARCReclaimReturnedObject:
    llvm_unreachable("NYI");
  case CK_ARCExtendBlockObject:
    llvm_unreachable("NYI");
  case CK_CopyAndAutoreleaseBlockObject:
    llvm_unreachable("NYI");
  case CK_FloatingRealToComplex:
    llvm_unreachable("NYI");
  case CK_FloatingComplexCast:
    llvm_unreachable("NYI");
  case CK_IntegralComplexToFloatingComplex:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToIntegralComplex:
    llvm_unreachable("NYI");
  case CK_ConstructorConversion:
    llvm_unreachable("NYI");
  case CK_ToUnion:
    llvm_unreachable("NYI");

  case CK_LValueToRValue:
    assert(CGF.getContext().hasSameUnqualifiedType(E->getType(), DestTy));
    assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr *>(E));

  case CK_IntegralToPointer:
    llvm_unreachable("NYI");
  case CK_PointerToIntegral:
    llvm_unreachable("NYI");
  case CK_ToVoid:
    llvm_unreachable("NYI");
  case CK_MatrixCast:
    llvm_unreachable("NYI");
  case CK_VectorSplat:
    llvm_unreachable("NYI");
  case CK_FixedPointCast:
    llvm_unreachable("NYI");
  case CK_FixedPointToBoolean:
    llvm_unreachable("NYI");
  case CK_FixedPointToIntegral:
    llvm_unreachable("NYI");
  case CK_IntegralToFixedPoint:
    llvm_unreachable("NYI");

  case CK_IntegralCast: {
    ScalarConversionOpts Opts;
    if (auto *ICE = dyn_cast<ImplicitCastExpr>(CE)) {
      if (!ICE->isPartOfExplicitCast())
        Opts = ScalarConversionOpts(CGF.SanOpts);
    }
    return buildScalarConversion(Visit(E), E->getType(), DestTy,
                                 CE->getExprLoc(), Opts);
  }

  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingCast:
  case CK_FixedPointToFloating:
  case CK_FloatingToFixedPoint: {
    if (Kind != CK_FloatingCast)
      llvm_unreachable("Only FloatingCast supported so far.");
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, CE);
    return buildScalarConversion(Visit(E), E->getType(), DestTy,
                                 CE->getExprLoc());
  }
  case CK_BooleanToSignedIntegral:
    llvm_unreachable("NYI");

  case CK_IntegralToBoolean: {
    return buildIntToBoolConversion(Visit(E), CGF.getLoc(CE->getSourceRange()));
  }

  case CK_PointerToBoolean:
    llvm_unreachable("NYI");
  case CK_FloatingToBoolean:
    llvm_unreachable("NYI");
  case CK_MemberPointerToBoolean:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToReal:
    llvm_unreachable("NYI");
  case CK_IntegralComplexToReal:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToBoolean:
    llvm_unreachable("NYI");
  case CK_IntegralComplexToBoolean:
    llvm_unreachable("NYI");
  case CK_ZeroToOCLOpaqueType:
    llvm_unreachable("NYI");
  case CK_IntToOCLSampler:
    llvm_unreachable("NYI");

  default:
    emitError(CGF.getLoc(CE->getExprLoc()), "cast kind not implemented: '")
        << CE->getCastKindName() << "'";
    return nullptr;
  } // end of switch

  llvm_unreachable("unknown scalar cast");
}

mlir::Value ScalarExprEmitter::VisitCallExpr(const CallExpr *E) {
  assert(!E->getCallReturnType(CGF.getContext())->isReferenceType() && "NYI");

  auto V = CGF.buildCallExpr(E).getScalarVal();

  // TODO: buildLValueAlignmentAssumption
  return V;
}

mlir::Value ScalarExprEmitter::VisitMemberExpr(MemberExpr *E) {
  // TODO(cir): Folding all this constants sound like work for MLIR optimizers,
  // keep assertion for now.
  assert(!UnimplementedFeature::tryEmitAsConstant());
  Expr::EvalResult Result;
  if (E->EvaluateAsInt(Result, CGF.getContext(), Expr::SE_AllowSideEffects))
    assert(0 && "NYI");
  return buildLoadOfLValue(E);
}

/// Emit a conversion from the specified type to the specified destination
/// type, both of which are CIR scalar types.
mlir::Value CIRGenFunction::buildScalarConversion(mlir::Value Src,
                                                  QualType SrcTy,
                                                  QualType DstTy,
                                                  SourceLocation Loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(SrcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this, builder)
      .buildScalarConversion(Src, SrcTy, DstTy, Loc);
}

/// If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the boolean result in Result.
bool CIRGenFunction::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                  bool &ResultBool,
                                                  bool AllowLabels) {
  llvm::APSInt ResultInt;
  if (!ConstantFoldsToSimpleInteger(Cond, ResultInt, AllowLabels))
    return false;

  ResultBool = ResultInt.getBoolValue();
  return true;
}

mlir::Value ScalarExprEmitter::VisitInitListExpr(InitListExpr *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();
  (void)Ignore;
  assert(Ignore == false && "init list ignored");
  unsigned NumInitElements = E->getNumInits();

  if (E->hadArrayRangeDesignator())
    llvm_unreachable("NYI");

  if (UnimplementedFeature::cirVectorType())
    llvm_unreachable("NYI");

  if (NumInitElements == 0) {
    // C++11 value-initialization for the scalar.
    llvm_unreachable("NYI");
  }

  return Visit(E->getInit(0));
}

mlir::Value ScalarExprEmitter::buildScalarCast(
    mlir::Value Src, QualType SrcType, QualType DstType, mlir::Type SrcTy,
    mlir::Type DstTy, ScalarConversionOpts Opts) {
  // The Element types determine the type of cast to perform.
  mlir::Type SrcElementTy;
  mlir::Type DstElementTy;
  QualType SrcElementType;
  QualType DstElementType;
  if (SrcType->isMatrixType() || DstType->isMatrixType()) {
    llvm_unreachable("NYI");
  } else {
    assert(!SrcType->isMatrixType() && !DstType->isMatrixType() &&
           "cannot cast between matrix and non-matrix types");
    SrcElementTy = SrcTy;
    DstElementTy = DstTy;
    SrcElementType = SrcType;
    DstElementType = DstType;
  }

  if (SrcElementTy.isa<mlir::IntegerType>()) {
    bool InputSigned = SrcElementType->isSignedIntegerOrEnumerationType();
    if (SrcElementType->isBooleanType() && Opts.TreatBooleanAsSigned) {
      llvm_unreachable("NYI");
    }

    if (DstElementTy.isa<mlir::IntegerType>())
      return Builder.create<mlir::cir::CastOp>(
          Src.getLoc(), DstTy, mlir::cir::CastKind::integral, Src);
    if (InputSigned)
      llvm_unreachable("NYI");

    llvm_unreachable("NYI");
  }

  if (DstElementTy.isa<mlir::IntegerType>()) {
    llvm_unreachable("NYI");
  }

  // if (DstElementTy.getTypeID() < SrcElementTy.getTypeID())
  //   llvm_unreachable("NYI");

  llvm_unreachable("NYI");
}

LValue
CIRGenFunction::buildCompoundAssignmentLValue(const CompoundAssignOperator *E) {
  ScalarExprEmitter Scalar(*this, builder);
  mlir::Value Result;
  switch (E->getOpcode()) {
#define COMPOUND_OP(Op)                                                        \
  case BO_##Op##Assign:                                                        \
    return Scalar.buildCompoundAssignLValue(E, &ScalarExprEmitter::build##Op,  \
                                            Result)
    COMPOUND_OP(Mul);
    COMPOUND_OP(Div);
    COMPOUND_OP(Rem);
    COMPOUND_OP(Add);
    COMPOUND_OP(Sub);
    COMPOUND_OP(Shl);
    COMPOUND_OP(Shr);
    COMPOUND_OP(And);
    COMPOUND_OP(Xor);
    COMPOUND_OP(Or);
#undef COMPOUND_OP

  case BO_PtrMemD:
  case BO_PtrMemI:
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Add:
  case BO_Sub:
  case BO_Shl:
  case BO_Shr:
  case BO_LT:
  case BO_GT:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_Cmp:
  case BO_And:
  case BO_Xor:
  case BO_Or:
  case BO_LAnd:
  case BO_LOr:
  case BO_Assign:
  case BO_Comma:
    llvm_unreachable("Not valid compound assignment operators");
  }
  llvm_unreachable("Unhandled compound assignment operator");
}

LValue ScalarExprEmitter::buildCompoundAssignLValue(
    const CompoundAssignOperator *E,
    mlir::Value (ScalarExprEmitter::*Func)(const BinOpInfo &),
    mlir::Value &Result) {
  QualType LHSTy = E->getLHS()->getType();
  BinOpInfo OpInfo;

  if (E->getComputationResultType()->isAnyComplexType())
    assert(0 && "not implemented");

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.
  OpInfo.RHS = Visit(E->getRHS());
  OpInfo.Ty = E->getComputationResultType();
  OpInfo.Opcode = E->getOpcode();
  OpInfo.FPFeatures = E->getFPFeaturesInEffect(CGF.getLangOpts());
  OpInfo.E = E;
  OpInfo.Loc = E->getSourceRange();

  // Load/convert the LHS
  LValue LHSLV = CGF.buildLValue(E->getLHS());

  if (const AtomicType *atomicTy = LHSTy->getAs<AtomicType>()) {
    assert(0 && "not implemented");
  }

  OpInfo.LHS = buildLoadOfLValue(LHSLV, E->getExprLoc());

  CIRGenFunction::SourceLocRAIIObject sourceloc{
      CGF, CGF.getLoc(E->getSourceRange())};
  SourceLocation Loc = E->getExprLoc();
  OpInfo.LHS =
      buildScalarConversion(OpInfo.LHS, LHSTy, E->getComputationLHSType(), Loc);

  // Expand the binary operator.
  Result = (this->*Func)(OpInfo);

  // Convert the result back to the LHS type,
  // potentially with Implicit Conversion sanitizer check.
  Result = buildScalarConversion(Result, E->getComputationResultType(), LHSTy,
                                 Loc, ScalarConversionOpts(CGF.SanOpts));

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (LHSLV.isBitField())
    assert(0 && "not yet implemented");
  else
    CGF.buildStoreThroughLValue(RValue::get(Result), LHSLV);

  assert(!CGF.getLangOpts().OpenMP && "Not implemented");
  return LHSLV;
}

mlir::Value ScalarExprEmitter::buildCompoundAssign(
    const CompoundAssignOperator *E,
    mlir::Value (ScalarExprEmitter::*Func)(const BinOpInfo &)) {

  bool Ignore = TestAndClearIgnoreResultAssign();
  mlir::Value RHS;
  LValue LHS = buildCompoundAssignLValue(E, Func, RHS);

  // If the result is clearly ignored, return now.
  if (Ignore)
    return {};

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return RHS;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LHS.isVolatile())
    return RHS;

  // Otherwise, reload the value.
  return buildLoadOfLValue(LHS, E->getExprLoc());
}
