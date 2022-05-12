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

#include "clang/AST/StmtVisitor.h"

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {

class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
  CIRGenFunction &CGF;
  mlir::OpBuilder &Builder;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, mlir::OpBuilder &builder,
                    bool ira = false)
      : CGF(cgf), Builder(builder) {}

  mlir::Value Visit(Expr *E) {
    return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
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

  // Handle l-values.
  mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
    // FIXME: we could try to emit this as constant first, see
    // CGF.tryEmitAsConstant(E)
    return buildLoadOfLValue(E);
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
  // Emit code for an explicit or implicit cast.  Implicit
  // casts have to handle a more broad range of conversions than explicit
  // casts, as they handle things like function to ptr-to-function decay
  // etc.
  mlir::Value VisitCastExpr(CastExpr *CE) {
    Expr *E = CE->getSubExpr();
    QualType DestTy = CE->getType();
    CastKind Kind = CE->getCastKind();
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
    case CK_NoOp:
      llvm_unreachable("NYI");
    case CK_BaseToDerived:
      llvm_unreachable("NYI");
    case CK_DerivedToBase:
      llvm_unreachable("NYI");
    case CK_Dynamic:
      llvm_unreachable("NYI");
    case CK_ArrayToPointerDecay:
      llvm_unreachable("NYI");
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

    case CK_IntegralToFloating:
      llvm_unreachable("NYI");
    case CK_FloatingToIntegral:
      llvm_unreachable("NYI");
    case CK_FloatingCast:
      llvm_unreachable("NYI");
    case CK_FixedPointToFloating:
      llvm_unreachable("NYI");
    case CK_FloatingToFixedPoint:
      llvm_unreachable("NYI");
    case CK_BooleanToSignedIntegral:
      llvm_unreachable("NYI");

    case CK_IntegralToBoolean: {
      return buildIntToBoolConversion(Visit(E),
                                      CGF.getLoc(CE->getSourceRange()));
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

  mlir::Value VisitCallExpr(const CallExpr *E) {
    assert(!E->getCallReturnType(CGF.getContext())->isReferenceType() && "NYI");

    auto V = CGF.buildCallExpr(E).getScalarVal();

    // TODO: buildLValueAlignmentAssumption
    return V;
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *E) {
    assert(!llvm::isa<MemberPointerType>(E->getType()) && "not implemented");
    return CGF.buildLValue(E->getSubExpr()).getPointer();
  }

  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty, Builder.getBoolAttr(E->getValue()));
  }

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

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    return build##OP(buildBinOps(E));                                          \
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

#define VISITCOMP(CODE)                                                        \
  mlir::Value VisitBin##CODE(const BinaryOperator *E) { return buildCmp(E); }
  VISITCOMP(LT)
  VISITCOMP(GT)
  VISITCOMP(LE)
  VISITCOMP(GE)
  VISITCOMP(EQ)
  VISITCOMP(NE)
#undef VISITCOMP

  mlir::Value VisitExpr(Expr *E) {
    // Crashing here for "ScalarExprClassName"? Please implement
    // VisitScalarExprClassName(...) to get this working.
    emitError(CGF.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
        << E->getStmtClassName() << "'";
    assert(0 && "shouldn't be here!");
    return {};
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
  mlir::Value buildScalarConversion(mlir::Value Src, QualType SrcType,
                                    QualType DstType, SourceLocation Loc) {
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

    // LLVM codegen ignore conversions like int -> uint, we should probably
    // emit it here in case lowering to sanitizers dialect at some point.
    if (SrcTy == DstTy) {
      assert(0 && "not implemented");
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
    assert(0 && "not implemented");
    mlir::Value Res = nullptr;
    mlir::Type ResTy = DstTy;

    // TODO: implement CGF.SanOpts.has(SanitizerKind::FloatCastOverflow)

    // Cast to half through float if half isn't a native type.
    if (DstType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      assert(0 && "not implemented");
    }

    // TODO: Res = EmitScalarCast(Src, SrcType, DstType, SrcTy, DstTy, Opts);
    if (DstTy != ResTy) {
      assert(0 && "not implemented");
    }

    return Res;
  }

  // Leaves.
  mlir::Value VisitIntegerLiteral(const IntegerLiteral *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        Builder.getIntegerAttr(Ty, E->getValue()));
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
