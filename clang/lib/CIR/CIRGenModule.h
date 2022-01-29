//===--- CIRGenModule.h - Per-Module state for CIR gen ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H

#include "CIRGenFunction.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Dialect/CIR/IR/CIRAttrs.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace cir {

/// Implementation of a CIR/MLIR emission from Clang AST.
///
/// This will emit operations that are specific to C(++)/ObjC(++) language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class CIRGenModule {
public:
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx);
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;
  ~CIRGenModule() = default;

  using SymTableTy = llvm::ScopedHashTable<const clang::Decl *, mlir::Value>;
  using SymTableScopeTy =
      llvm::ScopedHashTableScope<const clang::Decl *, mlir::Value>;

private:
  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated,
  /// the scope is destroyed and the mappings created in this scope are
  /// dropped.
  SymTableTy symbolTable;

  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  /// Per-function codegen information. Updated everytime buildCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *CurCGF = nullptr;

  /// Per-module type mapping from clang AST to CIR.
  std::unique_ptr<CIRGenTypes> genTypes;

  /// Use to track source locations across nested visitor traversals.
  /// Always use a `SourceLocRAIIObject` to change currSrcLoc.
  std::optional<mlir::Location> currSrcLoc;
  class SourceLocRAIIObject {
    CIRGenModule &P;
    std::optional<mlir::Location> OldVal;

  public:
    SourceLocRAIIObject(CIRGenModule &p, mlir::Location Value) : P(p) {
      if (P.currSrcLoc)
        OldVal = P.currSrcLoc;
      P.currSrcLoc = Value;
    }

    /// Can be used to restore the state early, before the dtor
    /// is run.
    void restore() { P.currSrcLoc = OldVal; }
    ~SourceLocRAIIObject() { restore(); }
  };

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(SourceLocation SLoc);

  mlir::Location getLoc(SourceRange SLoc);

  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const Decl *var, QualType T, mlir::Location loc,
                              CharUnits alignment, mlir::Value &addr,
                              bool IsParam = false);

public:
  mlir::ModuleOp getModule() { return theModule; }
  mlir::OpBuilder &getBuilder() { return builder; }

  class ScalarExprEmitter
      : public clang::StmtVisitor<ScalarExprEmitter, mlir::Value> {
    LLVM_ATTRIBUTE_UNUSED CIRGenFunction &CGF;
    CIRGenModule &CGM;

  public:
    ScalarExprEmitter(CIRGenFunction &cgf, CIRGenModule &cgm)
        : CGF(cgf), CGM(cgm) {}

    mlir::Value Visit(Expr *E) {
      return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
    }

    /// Emits the address of the l-value, then loads and returns the result.
    mlir::Value buildLoadOfLValue(const Expr *E) {
      LValue LV = CGM.buildLValue(E);
      auto load = CGM.builder.create<mlir::cir::LoadOp>(
          CGM.getLoc(E->getExprLoc()), CGM.getCIRType(E->getType()),
          LV.getPointer(), mlir::UnitAttr::get(CGM.builder.getContext()));
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
      clang::CastKind Kind = CE->getCastKind();
      switch (Kind) {
      case CK_LValueToRValue:
        assert(CGM.astCtx.hasSameUnqualifiedType(E->getType(), DestTy));
        assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
        return Visit(const_cast<Expr *>(E));
      case CK_NullToPointer: {
        // FIXME: use MustVisitNullValue(E) and evaluate expr.
        // Note that DestTy is used as the MLIR type instead of a custom
        // nullptr type.
        mlir::Type Ty = CGM.getCIRType(DestTy);
        return CGM.builder.create<mlir::cir::ConstantOp>(
            CGM.getLoc(E->getExprLoc()), Ty,
            mlir::cir::NullAttr::get(CGM.builder.getContext(), Ty));
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
      assert(!isa<MemberPointerType>(E->getType()) && "not implemented");
      return CGM.buildLValue(E->getSubExpr()).getPointer();
    }

    mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
      mlir::Type Ty = CGM.getCIRType(E->getType());
      return CGM.builder.create<mlir::cir::ConstantOp>(
          CGM.getLoc(E->getExprLoc()), Ty,
          CGM.builder.getBoolAttr(E->getValue()));
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
        if (const auto *BinOp = dyn_cast<BinaryOperator>(E)) {
          QualType LHSType = BinOp->getLHS()->getType();
          QualType RHSType = BinOp->getRHS()->getType();
          return LHSType->isFixedPointType() || RHSType->isFixedPointType();
        }
        if (const auto *UnOp = dyn_cast<UnaryOperator>(E))
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
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Mul, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildDiv(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Div, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildRem(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Rem, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildAdd(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Add, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildSub(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Sub, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildShl(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Shl, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildShr(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Shr, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildAnd(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::And, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildXor(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
          CGM.getLoc(Ops.Loc), CGM.getCIRType(Ops.Ty),
          mlir::cir::BinOpKind::Xor, Ops.LHS, Ops.RHS);
    }
    mlir::Value buildOr(const BinOpInfo &Ops) {
      return CGM.builder.create<mlir::cir::BinOp>(
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

          return CGM.builder.create<mlir::cir::CmpOp>(
              CGM.getLoc(BOInfo.Loc), CGM.getCIRType(BOInfo.Ty), Kind,
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

      return buildScalarConversion(Result, CGM.astCtx.BoolTy, E->getType(),
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
      emitError(CGM.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
          << E->getStmtClassName() << "'";
      assert(0 && "shouldn't be here!");
      return {};
    }

    mlir::Value buildIntToBoolConversion(mlir::Value srcVal,
                                         mlir::Location loc) {
      // Because of the type rules of C, we often end up computing a
      // logical value, then zero extending it to int, then wanting it
      // as a logical value again.
      // TODO: optimize this common case here or leave it for later
      // CIR passes?
      mlir::Type boolTy = CGM.getCIRType(CGM.astCtx.BoolTy);
      return CGM.builder.create<mlir::cir::CastOp>(
          loc, boolTy, mlir::cir::CastKind::int_to_bool, srcVal);
    }

    /// EmitConversionToBool - Convert the specified expression value to a
    /// boolean (i1) truth value.  This is equivalent to "Val != 0".
    mlir::Value buildConversionToBool(mlir::Value Src, QualType SrcType,
                                      mlir::Location loc) {
      assert(SrcType.isCanonical() && "EmitScalarConversion strips typedefs");

      if (SrcType->isRealFloatingType())
        assert(0 && "not implemented");

      if (const MemberPointerType *MPT = dyn_cast<MemberPointerType>(SrcType))
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

      SrcType = CGM.astCtx.getCanonicalType(SrcType);
      DstType = CGM.astCtx.getCanonicalType(DstType);
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
      if (SrcType->isHalfType() && !CGM.astCtx.getLangOpts().NativeHalfType) {
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
        // Must be an ptr to int cast.
        assert(DstTy.isa<mlir::IntegerType>() && "not ptr->int?");
        assert(0 && "not implemented");
      }

      // A scalar can be splatted to an extended vector of the same element type
      if (DstType->isExtVectorType() && !SrcType->isVectorType()) {
        // Sema should add casts to make sure that the source expression's type
        // is the same as the vector's element type (sans qualifiers)
        assert(
            DstType->castAs<ExtVectorType>()->getElementType().getTypePtr() ==
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
      if (DstType->isHalfType() && !CGM.astCtx.getLangOpts().NativeHalfType) {
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
      return CGM.builder.create<mlir::cir::ConstantOp>(
          CGM.getLoc(E->getExprLoc()), Ty,
          CGM.builder.getIntegerAttr(Ty, E->getValue()));
    }
  };

  struct AutoVarEmission {
    const VarDecl *Variable;
    /// The address of the alloca for languages with explicit address space
    /// (e.g. OpenCL) or alloca casted to generic pointer for address space
    /// agnostic languages (e.g. C++). Invalid if the variable was emitted
    /// as a global constant.
    Address Addr;

    /// True if the variable is of aggregate type and has a constant
    /// initializer.
    bool IsConstantAggregate;

    struct Invalid {};
    AutoVarEmission(Invalid) : Variable(nullptr), Addr(Address::invalid()) {}

    AutoVarEmission(const VarDecl &variable)
        : Variable(&variable), Addr(Address::invalid()),
          IsConstantAggregate(false) {}

    static AutoVarEmission invalid() { return AutoVarEmission(Invalid()); }
    /// Returns the raw, allocated address, which is not necessarily
    /// the address of the object itself. It is casted to default
    /// address space for address space agnostic languages.
    Address getAllocatedAddress() const { return Addr; }
  };

  /// Determine whether an object of this type can be emitted
  /// as a constant.
  ///
  /// If ExcludeCtor is true, the duration when the object's constructor runs
  /// will not be considered. The caller will need to verify that the object is
  /// not written to during its construction.
  /// FIXME: in LLVM codegen path this is part of CGM, which doesn't seem
  /// like necessary, since (1) it doesn't use CGM at all and (2) is AST type
  /// query specific.
  bool isTypeConstant(QualType Ty, bool ExcludeCtor);

  /// Emit the alloca and debug information for a
  /// local variable.  Does not emit initialization or destruction.
  AutoVarEmission buildAutoVarAlloca(const VarDecl &D);

  /// Determine whether the given initializer is trivial in the sense
  /// that it requires no code to be generated.
  bool isTrivialInitializer(const Expr *Init);

  // TODO: this can also be abstrated into common AST helpers
  bool hasBooleanRepresentation(QualType Ty);

  mlir::Value buildToMemory(mlir::Value Value, QualType Ty);

  void buildStoreOfScalar(mlir::Value value, LValue lvalue,
                          const Decl *InitDecl);

  void buildStoreOfScalar(mlir::Value Value, Address Addr, bool Volatile,
                          QualType Ty, LValueBaseInfo BaseInfo,
                          const Decl *InitDecl, bool isNontemporal);

  /// Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void buldStoreThroughLValue(RValue Src, LValue Dst, const Decl *InitDecl);

  void buildScalarInit(const Expr *init, const ValueDecl *D, LValue lvalue);

  /// Emit an expression as an initializer for an object (variable, field, etc.)
  /// at the given location.  The expression is not necessarily the normal
  /// initializer for the object, and the address is not necessarily
  /// its normal location.
  ///
  /// \param init the initializing expression
  /// \param D the object to act as if we're initializing
  /// \param lvalue the lvalue to initialize
  void buildExprAsInit(const Expr *init, const ValueDecl *D, LValue lvalue);

  void buildAutoVarInit(const AutoVarEmission &emission);

  void buildAutoVarCleanups(const AutoVarEmission &emission);

  /// Emit code and set up symbol table for a variable declaration with auto,
  /// register, or no storage class specifier. These turn into simple stack
  /// objects, globals depending on target.
  void buildAutoVarDecl(const VarDecl &D);

  /// This method handles emission of any variable declaration
  /// inside a function, including static vars etc.
  void buildVarDecl(const VarDecl &D);

  void buildDecl(const Decl &D);

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value buildScalarExpr(const Expr *E);

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  mlir::Value buildScalarConversion(mlir::Value Src, QualType SrcTy,
                                    QualType DstTy, SourceLocation Loc);

  mlir::LogicalResult buildReturnStmt(const ReturnStmt &S);

  mlir::LogicalResult buildDeclStmt(const DeclStmt &S);

  mlir::LogicalResult buildSimpleStmt(const Stmt *S, bool useCurrentScope);

  LValue buildDeclRefLValue(const DeclRefExpr *E);

  /// Emit code to compute the specified expression which
  /// can have any type.  The result is returned as an RValue struct.
  /// TODO: if this is an aggregate expression, add a AggValueSlot to indicate
  /// where the result should be returned.
  RValue buildAnyExpr(const Expr *E);

  LValue buildBinaryOperatorLValue(const BinaryOperator *E);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// Return the best known alignment for an unknown pointer to a
  /// particular class.
  CharUnits getClassPointerAlignment(const CXXRecordDecl *RD);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  CharUnits getNaturalPointeeTypeAlignment(QualType T,
                                           LValueBaseInfo *BaseInfo);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// TODO: Add TBAAAccessInfo
  CharUnits getNaturalTypeAlignment(QualType T, LValueBaseInfo *BaseInfo,
                                    bool forPointeeType);

  /// Given an expression of pointer type, try to
  /// derive a more accurate bound on the alignment of the pointer.
  Address buildPointerWithAlignment(const Expr *E, LValueBaseInfo *BaseInfo);

  LValue buildUnaryOpLValue(const UnaryOperator *E);

  /// Emit code to compute a designator that specifies the location
  /// of the expression.
  /// FIXME: document this function better.
  LValue buildLValue(const Expr *E);

  /// EmitIgnoredExpr - Emit code to compute the specified expression,
  /// ignoring the result.
  void buildIgnoredExpr(const Expr *E);

  /// If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the boolean result in Result.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, bool &ResultBool,
                                    bool AllowLabels);

  /// Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  bool ContainsLabel(const Stmt *S, bool IgnoreCaseStmts = false);

  /// If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return false.  If it
  /// constant folds return true and set the folded value.
  bool ConstantFoldsToSimpleInteger(const Expr *Cond, llvm::APSInt &ResultInt,
                                    bool AllowLabels);

  /// Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  mlir::Value evaluateExprAsBool(const Expr *E);

  /// Emit an if on a boolean condition to the specified blocks.
  /// FIXME: Based on the condition, this might try to simplify the codegen of
  /// the conditional based on the branch. TrueCount should be the number of
  /// times we expect the condition to evaluate to true based on PGO data. We
  /// might decide to leave this as a separate pass (see EmitBranchOnBoolExpr
  /// for extra ideas).
  mlir::LogicalResult buildIfOnBoolExpr(const Expr *cond, mlir::Location loc,
                                        const Stmt *thenS, const Stmt *elseS);

  mlir::LogicalResult buildIfStmt(const IfStmt &S);

  // Build CIR for a statement. useCurrentScope should be true if no
  // new scopes need be created when finding a compound statement.
  mlir::LogicalResult buildStmt(const Stmt *S, bool useCurrentScope);

  mlir::LogicalResult buildFunctionBody(const Stmt *Body);

  mlir::LogicalResult buildCompoundStmt(const CompoundStmt &S);

  mlir::LogicalResult buildCompoundStmtWithoutScope(const CompoundStmt &S);

  void buildTopLevelDecl(Decl *decl);

  // Emit a new function and add it to the MLIR module.
  mlir::FuncOp buildFunction(const FunctionDecl *FD);

  mlir::Type getCIRType(const QualType &type);

  void verifyModule();
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
