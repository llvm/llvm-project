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
  LLVM_ATTRIBUTE_UNUSED CIRGenFunction &CGF;
  CIRGenModule &CGM;
  mlir::OpBuilder &Builder;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, CIRGenModule &cgm,
                    mlir::OpBuilder &builder)
      : CGF(cgf), CGM(cgm), Builder(builder) {}

  mlir::Value Visit(Expr *E) {
    return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
  }

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value buildLoadOfLValue(const Expr *E) {
    LValue LV = CGM.buildLValue(E);
    auto load = Builder.create<mlir::cir::LoadOp>(
        CGM.getLoc(E->getExprLoc()), CGM.getCIRType(E->getType()),
        LV.getPointer(), mlir::UnitAttr::get(Builder.getContext()));
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return load;
  }

  // Handle l-values.
  mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
    // FIXME: we could try to emit this as constant first, see
    // CGF.tryEmitAsConstant(E)
    return buildLoadOfLValue(E);
  }

  // Emit code for an explicit or implicit cast.  Implicit
  // casts have to handle a more broad range of conversions than explicit
  // casts, as they handle things like function to ptr-to-function decay
  // etc.
  mlir::Value VisitCastExpr(CastExpr *CE) {
    Expr *E = CE->getSubExpr();
    QualType DestTy = CE->getType();
    CastKind Kind = CE->getCastKind();
    switch (Kind) {
    case CK_LValueToRValue:
      assert(CGM.getASTContext().hasSameUnqualifiedType(E->getType(), DestTy));
      assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
      return Visit(const_cast<Expr *>(E));
    case CK_NullToPointer: {
      // FIXME: use MustVisitNullValue(E) and evaluate expr.
      // Note that DestTy is used as the MLIR type instead of a custom
      // nullptr type.
      mlir::Type Ty = CGM.getCIRType(DestTy);
      return Builder.create<mlir::cir::ConstantOp>(
          CGM.getLoc(E->getExprLoc()), Ty,
          mlir::cir::NullAttr::get(Builder.getContext(), Ty));
    }
    case CK_IntegralToBoolean: {
      return buildIntToBoolConversion(Visit(E),
                                      CGM.getLoc(CE->getSourceRange()));
    }
    default:
      emitError(CGM.getLoc(CE->getExprLoc()), "cast kind not implemented: '")
          << CE->getCastKindName() << "'";
      assert(0 && "not implemented");
      return nullptr;
    }
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *E) {
    assert(!llvm::isa<MemberPointerType>(E->getType()) && "not implemented");
    return CGM.buildLValue(E->getSubExpr()).getPointer();
  }

  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    mlir::Type Ty = CGM.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGM.getLoc(E->getExprLoc()), Ty, Builder.getBoolAttr(E->getValue()));
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
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Mul,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildDiv(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Div,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildRem(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Rem,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildAdd(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Add,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildSub(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Sub,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildShl(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Shl,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildShr(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Shr,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildAnd(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::And,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildXor(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Xor,
        Ops.LHS, Ops.RHS);
  }
  mlir::Value buildOr(const BinOpInfo &Ops) {
    return Builder.create<mlir::cir::BinOp>(
        CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Or,
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

        return Builder.create<mlir::cir::CmpOp>(CGM.getLoc(BOInfo.Loc),
                                                CGM.getCIRType(BOInfo.Ty), Kind,
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

    return buildScalarConversion(Result, CGM.getASTContext().BoolTy,
                                 E->getType(), E->getExprLoc());
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
    emitError(CGM.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
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
    mlir::Type boolTy = CGM.getCIRType(CGM.getASTContext().BoolTy);
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

    SrcType = CGM.getASTContext().getCanonicalType(SrcType);
    DstType = CGM.getASTContext().getCanonicalType(DstType);
    if (SrcType == DstType)
      return Src;

    if (DstType->isVoidType())
      return nullptr;
    mlir::Type SrcTy = Src.getType();

    // Handle conversions to bool first, they are special: comparisons against
    // 0.
    if (DstType->isBooleanType())
      return buildConversionToBool(Src, SrcType, CGM.getLoc(Loc));

    mlir::Type DstTy = CGM.getCIRType(DstType);

    // Cast from half through float if half isn't a native type.
    if (SrcType->isHalfType() &&
        !CGM.getASTContext().getLangOpts().NativeHalfType) {
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
        !CGM.getASTContext().getLangOpts().NativeHalfType) {
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
    mlir::Type Ty = CGM.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGM.getLoc(E->getExprLoc()), Ty,
        Builder.getIntegerAttr(Ty, E->getValue()));
  }
};

} // namespace

/// Emit the computation of the specified expression of scalar type,
/// ignoring the result.
mlir::Value CIRGenModule::buildScalarExpr(const Expr *E) {
  assert(E && CIRGenFunction::hasScalarEvaluationKind(E->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*CurCGF, *this, builder)
      .Visit(const_cast<Expr *>(E));
}

/// Emit a conversion from the specified type to the specified destination
/// type, both of which are CIR scalar types.
mlir::Value CIRGenModule::buildScalarConversion(mlir::Value Src, QualType SrcTy,
                                                QualType DstTy,
                                                SourceLocation Loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(SrcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*CurCGF, *this, builder)
      .buildScalarConversion(Src, SrcTy, DstTy, Loc);
}
