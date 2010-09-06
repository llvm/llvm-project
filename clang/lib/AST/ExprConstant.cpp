//===--- ExprConstant.cpp - Expression Constant Evaluator -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr constant evaluator.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include <cstring>

using namespace clang;
using llvm::APSInt;
using llvm::APFloat;

/// EvalInfo - This is a private struct used by the evaluator to capture
/// information about a subexpression as it is folded.  It retains information
/// about the AST context, but also maintains information about the folded
/// expression.
///
/// If an expression could be evaluated, it is still possible it is not a C
/// "integer constant expression" or constant expression.  If not, this struct
/// captures information about how and why not.
///
/// One bit of information passed *into* the request for constant folding
/// indicates whether the subexpression is "evaluated" or not according to C
/// rules.  For example, the RHS of (0 && foo()) is not evaluated.  We can
/// evaluate the expression regardless of what the RHS is, but C only allows
/// certain things in certain situations.
struct EvalInfo {
  ASTContext &Ctx;

  /// EvalResult - Contains information about the evaluation.
  Expr::EvalResult &EvalResult;

  EvalInfo(ASTContext &ctx, Expr::EvalResult& evalresult)
    : Ctx(ctx), EvalResult(evalresult) {}
};

namespace {
  struct ComplexValue {
  private:
    bool IsInt;

  public:
    APSInt IntReal, IntImag;
    APFloat FloatReal, FloatImag;

    ComplexValue() : FloatReal(APFloat::Bogus), FloatImag(APFloat::Bogus) {}

    void makeComplexFloat() { IsInt = false; }
    bool isComplexFloat() const { return !IsInt; }
    APFloat &getComplexFloatReal() { return FloatReal; }
    APFloat &getComplexFloatImag() { return FloatImag; }

    void makeComplexInt() { IsInt = true; }
    bool isComplexInt() const { return IsInt; }
    APSInt &getComplexIntReal() { return IntReal; }
    APSInt &getComplexIntImag() { return IntImag; }

    void moveInto(APValue &v) {
      if (isComplexFloat())
        v = APValue(FloatReal, FloatImag);
      else
        v = APValue(IntReal, IntImag);
    }
  };

  struct LValue {
    Expr *Base;
    CharUnits Offset;

    Expr *getLValueBase() { return Base; }
    CharUnits getLValueOffset() { return Offset; }

    void moveInto(APValue &v) {
      v = APValue(Base, Offset);
    }
  };
}

static bool EvaluateLValue(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluatePointer(const Expr *E, LValue &Result, EvalInfo &Info);
static bool EvaluateInteger(const Expr *E, APSInt  &Result, EvalInfo &Info);
static bool EvaluateIntegerOrLValue(const Expr *E, APValue  &Result,
                                    EvalInfo &Info);
static bool EvaluateFloat(const Expr *E, APFloat &Result, EvalInfo &Info);
static bool EvaluateComplex(const Expr *E, ComplexValue &Res, EvalInfo &Info);

//===----------------------------------------------------------------------===//
// Misc utilities
//===----------------------------------------------------------------------===//

static bool IsGlobalLValue(const Expr* E) {
  if (!E) return true;

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (isa<FunctionDecl>(DRE->getDecl()))
      return true;
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return VD->hasGlobalStorage();
    return false;
  }

  if (const CompoundLiteralExpr *CLE = dyn_cast<CompoundLiteralExpr>(E))
    return CLE->isFileScope();

  return true;
}

static bool EvalPointerValueAsBool(LValue& Value, bool& Result) {
  const Expr* Base = Value.Base;

  // A null base expression indicates a null pointer.  These are always
  // evaluatable, and they are false unless the offset is zero.
  if (!Base) {
    Result = !Value.Offset.isZero();
    return true;
  }

  // Require the base expression to be a global l-value.
  if (!IsGlobalLValue(Base)) return false;

  // We have a non-null base expression.  These are generally known to
  // be true, but if it'a decl-ref to a weak symbol it can be null at
  // runtime.
  Result = true;

  const DeclRefExpr* DeclRef = dyn_cast<DeclRefExpr>(Base);
  if (!DeclRef)
    return true;

  // If it's a weak symbol, it isn't constant-evaluable.
  const ValueDecl* Decl = DeclRef->getDecl();
  if (Decl->hasAttr<WeakAttr>() ||
      Decl->hasAttr<WeakRefAttr>() ||
      Decl->hasAttr<WeakImportAttr>())
    return false;

  return true;
}

static bool HandleConversionToBool(const Expr* E, bool& Result,
                                   EvalInfo &Info) {
  if (E->getType()->isIntegralOrEnumerationType()) {
    APSInt IntResult;
    if (!EvaluateInteger(E, IntResult, Info))
      return false;
    Result = IntResult != 0;
    return true;
  } else if (E->getType()->isRealFloatingType()) {
    APFloat FloatResult(0.0);
    if (!EvaluateFloat(E, FloatResult, Info))
      return false;
    Result = !FloatResult.isZero();
    return true;
  } else if (E->getType()->hasPointerRepresentation()) {
    LValue PointerResult;
    if (!EvaluatePointer(E, PointerResult, Info))
      return false;
    return EvalPointerValueAsBool(PointerResult, Result);
  } else if (E->getType()->isAnyComplexType()) {
    ComplexValue ComplexResult;
    if (!EvaluateComplex(E, ComplexResult, Info))
      return false;
    if (ComplexResult.isComplexFloat()) {
      Result = !ComplexResult.getComplexFloatReal().isZero() ||
               !ComplexResult.getComplexFloatImag().isZero();
    } else {
      Result = ComplexResult.getComplexIntReal().getBoolValue() ||
               ComplexResult.getComplexIntImag().getBoolValue();
    }
    return true;
  }

  return false;
}

static APSInt HandleFloatToIntCast(QualType DestType, QualType SrcType,
                                   APFloat &Value, ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  // Determine whether we are converting to unsigned or signed.
  bool DestSigned = DestType->isSignedIntegerType();

  // FIXME: Warning for overflow.
  uint64_t Space[4];
  bool ignored;
  (void)Value.convertToInteger(Space, DestWidth, DestSigned,
                               llvm::APFloat::rmTowardZero, &ignored);
  return APSInt(llvm::APInt(DestWidth, 4, Space), !DestSigned);
}

static APFloat HandleFloatToFloatCast(QualType DestType, QualType SrcType,
                                      APFloat &Value, ASTContext &Ctx) {
  bool ignored;
  APFloat Result = Value;
  Result.convert(Ctx.getFloatTypeSemantics(DestType),
                 APFloat::rmNearestTiesToEven, &ignored);
  return Result;
}

static APSInt HandleIntToIntCast(QualType DestType, QualType SrcType,
                                 APSInt &Value, ASTContext &Ctx) {
  unsigned DestWidth = Ctx.getIntWidth(DestType);
  APSInt Result = Value;
  // Figure out if this is a truncate, extend or noop cast.
  // If the input is signed, do a sign extend, noop, or truncate.
  Result.extOrTrunc(DestWidth);
  Result.setIsUnsigned(DestType->isUnsignedIntegerType());
  return Result;
}

static APFloat HandleIntToFloatCast(QualType DestType, QualType SrcType,
                                    APSInt &Value, ASTContext &Ctx) {

  APFloat Result(Ctx.getFloatTypeSemantics(DestType), 1);
  Result.convertFromAPInt(Value, Value.isSigned(),
                          APFloat::rmNearestTiesToEven);
  return Result;
}

namespace {
class HasSideEffect
  : public StmtVisitor<HasSideEffect, bool> {
  EvalInfo &Info;
public:

  HasSideEffect(EvalInfo &info) : Info(info) {}

  // Unhandled nodes conservatively default to having side effects.
  bool VisitStmt(Stmt *S) {
    return true;
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return false;
  }
  // We don't want to evaluate BlockExprs multiple times, as they generate
  // a ton of code.
  bool VisitBlockExpr(BlockExpr *E) { return true; }
  bool VisitPredefinedExpr(PredefinedExpr *E) { return false; }
  bool VisitCompoundLiteralExpr(CompoundLiteralExpr *E)
    { return Visit(E->getInitializer()); }
  bool VisitMemberExpr(MemberExpr *E) { return Visit(E->getBase()); }
  bool VisitIntegerLiteral(IntegerLiteral *E) { return false; }
  bool VisitFloatingLiteral(FloatingLiteral *E) { return false; }
  bool VisitStringLiteral(StringLiteral *E) { return false; }
  bool VisitCharacterLiteral(CharacterLiteral *E) { return false; }
  bool VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) { return false; }
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *E)
    { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitChooseExpr(ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitCastExpr(CastExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitBinAssign(BinaryOperator *E) { return true; }
  bool VisitCompoundAssignOperator(BinaryOperator *E) { return true; }
  bool VisitBinaryOperator(BinaryOperator *E)
  { return Visit(E->getLHS()) || Visit(E->getRHS()); }
  bool VisitUnaryPreInc(UnaryOperator *E) { return true; }
  bool VisitUnaryPostInc(UnaryOperator *E) { return true; }
  bool VisitUnaryPreDec(UnaryOperator *E) { return true; }
  bool VisitUnaryPostDec(UnaryOperator *E) { return true; }
  bool VisitUnaryDeref(UnaryOperator *E) {
    if (Info.Ctx.getCanonicalType(E->getType()).isVolatileQualified())
      return true;
    return Visit(E->getSubExpr());
  }
  bool VisitUnaryOperator(UnaryOperator *E) { return Visit(E->getSubExpr()); }
    
  // Has side effects if any element does.
  bool VisitInitListExpr(InitListExpr *E) {
    for (unsigned i = 0, e = E->getNumInits(); i != e; ++i)
      if (Visit(E->getInit(i))) return true;
    return false;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LValue Evaluation
//===----------------------------------------------------------------------===//
namespace {
class LValueExprEvaluator
  : public StmtVisitor<LValueExprEvaluator, bool> {
  EvalInfo &Info;
  LValue &Result;

  bool Success(Expr *E) {
    Result.Base = E;
    Result.Offset = CharUnits::Zero();
    return true;
  }
public:

  LValueExprEvaluator(EvalInfo &info, LValue &Result) :
    Info(info), Result(Result) {}

  bool VisitStmt(Stmt *S) {
    return false;
  }
  
  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitDeclRefExpr(DeclRefExpr *E);
  bool VisitPredefinedExpr(PredefinedExpr *E) { return Success(E); }
  bool VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  bool VisitMemberExpr(MemberExpr *E);
  bool VisitStringLiteral(StringLiteral *E) { return Success(E); }
  bool VisitObjCEncodeExpr(ObjCEncodeExpr *E) { return Success(E); }
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *E);
  bool VisitUnaryDeref(UnaryOperator *E);
  bool VisitUnaryExtension(const UnaryOperator *E)
    { return Visit(E->getSubExpr()); }
  bool VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }

  bool VisitCastExpr(CastExpr *E) {
    switch (E->getCastKind()) {
    default:
      return false;

    case CK_NoOp:
      return Visit(E->getSubExpr());
    }
  }
  // FIXME: Missing: __real__, __imag__
};
} // end anonymous namespace

static bool EvaluateLValue(const Expr* E, LValue& Result, EvalInfo &Info) {
  return LValueExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool LValueExprEvaluator::VisitDeclRefExpr(DeclRefExpr *E) {
  if (isa<FunctionDecl>(E->getDecl())) {
    return Success(E);
  } else if (VarDecl* VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (!VD->getType()->isReferenceType())
      return Success(E);
    // Reference parameters can refer to anything even if they have an
    // "initializer" in the form of a default argument.
    if (isa<ParmVarDecl>(VD))
      return false;
    // FIXME: Check whether VD might be overridden!
    if (const Expr *Init = VD->getAnyInitializer())
      return Visit(const_cast<Expr *>(Init));
  }

  return false;
}

bool LValueExprEvaluator::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  return Success(E);
}

bool LValueExprEvaluator::VisitMemberExpr(MemberExpr *E) {
  QualType Ty;
  if (E->isArrow()) {
    if (!EvaluatePointer(E->getBase(), Result, Info))
      return false;
    Ty = E->getBase()->getType()->getAs<PointerType>()->getPointeeType();
  } else {
    if (!Visit(E->getBase()))
      return false;
    Ty = E->getBase()->getType();
  }

  RecordDecl *RD = Ty->getAs<RecordType>()->getDecl();
  const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

  FieldDecl *FD = dyn_cast<FieldDecl>(E->getMemberDecl());
  if (!FD) // FIXME: deal with other kinds of member expressions
    return false;

  if (FD->getType()->isReferenceType())
    return false;

  // FIXME: This is linear time.
  unsigned i = 0;
  for (RecordDecl::field_iterator Field = RD->field_begin(),
                               FieldEnd = RD->field_end();
       Field != FieldEnd; (void)++Field, ++i) {
    if (*Field == FD)
      break;
  }

  Result.Offset += CharUnits::fromQuantity(RL.getFieldOffset(i) / 8);
  return true;
}

bool LValueExprEvaluator::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  if (!EvaluatePointer(E->getBase(), Result, Info))
    return false;

  APSInt Index;
  if (!EvaluateInteger(E->getIdx(), Index, Info))
    return false;

  CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(E->getType());
  Result.Offset += Index.getSExtValue() * ElementSize;
  return true;
}

bool LValueExprEvaluator::VisitUnaryDeref(UnaryOperator *E) {
  return EvaluatePointer(E->getSubExpr(), Result, Info);
}

//===----------------------------------------------------------------------===//
// Pointer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class PointerExprEvaluator
  : public StmtVisitor<PointerExprEvaluator, bool> {
  EvalInfo &Info;
  LValue &Result;

  bool Success(Expr *E) {
    Result.Base = E;
    Result.Offset = CharUnits::Zero();
    return true;
  }
public:

  PointerExprEvaluator(EvalInfo &info, LValue &Result)
    : Info(info), Result(Result) {}

  bool VisitStmt(Stmt *S) {
    return false;
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitCastExpr(CastExpr* E);
  bool VisitUnaryExtension(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
  bool VisitUnaryAddrOf(const UnaryOperator *E);
  bool VisitObjCStringLiteral(ObjCStringLiteral *E)
      { return Success(E); }
  bool VisitAddrLabelExpr(AddrLabelExpr *E)
      { return Success(E); }
  bool VisitCallExpr(CallExpr *E);
  bool VisitBlockExpr(BlockExpr *E) {
    if (!E->hasBlockDeclRefExprs())
      return Success(E);
    return false;
  }
  bool VisitImplicitValueInitExpr(ImplicitValueInitExpr *E)
      { return Success((Expr*)0); }
  bool VisitConditionalOperator(ConditionalOperator *E);
  bool VisitChooseExpr(ChooseExpr *E)
      { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E)
      { return Success((Expr*)0); }
  // FIXME: Missing: @protocol, @selector
};
} // end anonymous namespace

static bool EvaluatePointer(const Expr* E, LValue& Result, EvalInfo &Info) {
  assert(E->getType()->hasPointerRepresentation());
  return PointerExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool PointerExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() != BO_Add &&
      E->getOpcode() != BO_Sub)
    return false;

  const Expr *PExp = E->getLHS();
  const Expr *IExp = E->getRHS();
  if (IExp->getType()->isPointerType())
    std::swap(PExp, IExp);

  if (!EvaluatePointer(PExp, Result, Info))
    return false;

  llvm::APSInt Offset;
  if (!EvaluateInteger(IExp, Offset, Info))
    return false;
  int64_t AdditionalOffset
    = Offset.isSigned() ? Offset.getSExtValue()
                        : static_cast<int64_t>(Offset.getZExtValue());

  // Compute the new offset in the appropriate width.

  QualType PointeeType =
    PExp->getType()->getAs<PointerType>()->getPointeeType();
  CharUnits SizeOfPointee;

  // Explicitly handle GNU void* and function pointer arithmetic extensions.
  if (PointeeType->isVoidType() || PointeeType->isFunctionType())
    SizeOfPointee = CharUnits::One();
  else
    SizeOfPointee = Info.Ctx.getTypeSizeInChars(PointeeType);

  if (E->getOpcode() == BO_Add)
    Result.Offset += AdditionalOffset * SizeOfPointee;
  else
    Result.Offset -= AdditionalOffset * SizeOfPointee;

  return true;
}

bool PointerExprEvaluator::VisitUnaryAddrOf(const UnaryOperator *E) {
  return EvaluateLValue(E->getSubExpr(), Result, Info);
}


bool PointerExprEvaluator::VisitCastExpr(CastExpr* E) {
  Expr* SubExpr = E->getSubExpr();

  switch (E->getCastKind()) {
  default:
    break;

  case CK_Unknown: {
    // FIXME: The handling for CK_Unknown is ugly/shouldn't be necessary!

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType() ||
        SubExpr->getType()->isObjCObjectPointerType() ||
        SubExpr->getType()->isNullPtrType() ||
        SubExpr->getType()->isBlockPointerType())
      return Visit(SubExpr);

    if (SubExpr->getType()->isIntegralOrEnumerationType()) {
      APValue Value;
      if (!EvaluateIntegerOrLValue(SubExpr, Value, Info))
        break;

      if (Value.isInt()) {
        Value.getInt().extOrTrunc((unsigned)Info.Ctx.getTypeSize(E->getType()));
        Result.Base = 0;
        Result.Offset = CharUnits::fromQuantity(Value.getInt().getZExtValue());
        return true;
      } else {
        Result.Base = Value.getLValueBase();
        Result.Offset = Value.getLValueOffset();
        return true;
      }
    }
    break;
  }

  case CK_NoOp:
  case CK_BitCast:
  case CK_LValueBitCast:
  case CK_AnyPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
    return Visit(SubExpr);

  case CK_IntegralToPointer: {
    APValue Value;
    if (!EvaluateIntegerOrLValue(SubExpr, Value, Info))
      break;

    if (Value.isInt()) {
      Value.getInt().extOrTrunc((unsigned)Info.Ctx.getTypeSize(E->getType()));
      Result.Base = 0;
      Result.Offset = CharUnits::fromQuantity(Value.getInt().getZExtValue());
      return true;
    } else {
      // Cast is of an lvalue, no need to change value.
      Result.Base = Value.getLValueBase();
      Result.Offset = Value.getLValueOffset();
      return true;
    }
  }
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
    return EvaluateLValue(SubExpr, Result, Info);
  }

  return false;
}

bool PointerExprEvaluator::VisitCallExpr(CallExpr *E) {
  if (E->isBuiltinCall(Info.Ctx) ==
        Builtin::BI__builtin___CFStringMakeConstantString ||
      E->isBuiltinCall(Info.Ctx) ==
        Builtin::BI__builtin___NSStringMakeConstantString)
    return Success(E);
  return false;
}

bool PointerExprEvaluator::VisitConditionalOperator(ConditionalOperator *E) {
  bool BoolResult;
  if (!HandleConversionToBool(E->getCond(), BoolResult, Info))
    return false;

  Expr* EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();
  return Visit(EvalExpr);
}

//===----------------------------------------------------------------------===//
// Vector Evaluation
//===----------------------------------------------------------------------===//

namespace {
  class VectorExprEvaluator
  : public StmtVisitor<VectorExprEvaluator, APValue> {
    EvalInfo &Info;
    APValue GetZeroVector(QualType VecType);
  public:

    VectorExprEvaluator(EvalInfo &info) : Info(info) {}

    APValue VisitStmt(Stmt *S) {
      return APValue();
    }

    APValue VisitParenExpr(ParenExpr *E)
        { return Visit(E->getSubExpr()); }
    APValue VisitUnaryExtension(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitUnaryPlus(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitUnaryReal(const UnaryOperator *E)
      { return Visit(E->getSubExpr()); }
    APValue VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E)
      { return GetZeroVector(E->getType()); }
    APValue VisitCastExpr(const CastExpr* E);
    APValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
    APValue VisitInitListExpr(const InitListExpr *E);
    APValue VisitConditionalOperator(const ConditionalOperator *E);
    APValue VisitChooseExpr(const ChooseExpr *E)
      { return Visit(E->getChosenSubExpr(Info.Ctx)); }
    APValue VisitUnaryImag(const UnaryOperator *E);
    // FIXME: Missing: unary -, unary ~, binary add/sub/mul/div,
    //                 binary comparisons, binary and/or/xor,
    //                 shufflevector, ExtVectorElementExpr
    //        (Note that these require implementing conversions
    //         between vector types.)
  };
} // end anonymous namespace

static bool EvaluateVector(const Expr* E, APValue& Result, EvalInfo &Info) {
  if (!E->getType()->isVectorType())
    return false;
  Result = VectorExprEvaluator(Info).Visit(const_cast<Expr*>(E));
  return !Result.isUninit();
}

APValue VectorExprEvaluator::VisitCastExpr(const CastExpr* E) {
  const VectorType *VTy = E->getType()->getAs<VectorType>();
  QualType EltTy = VTy->getElementType();
  unsigned NElts = VTy->getNumElements();
  unsigned EltWidth = Info.Ctx.getTypeSize(EltTy);

  const Expr* SE = E->getSubExpr();
  QualType SETy = SE->getType();
  APValue Result = APValue();

  // Check for vector->vector bitcast and scalar->vector splat.
  if (SETy->isVectorType()) {
    return this->Visit(const_cast<Expr*>(SE));
  } else if (SETy->isIntegerType()) {
    APSInt IntResult;
    if (!EvaluateInteger(SE, IntResult, Info))
      return APValue();
    Result = APValue(IntResult);
  } else if (SETy->isRealFloatingType()) {
    APFloat F(0.0);
    if (!EvaluateFloat(SE, F, Info))
      return APValue();
    Result = APValue(F);
  } else
    return APValue();

  // For casts of a scalar to ExtVector, convert the scalar to the element type
  // and splat it to all elements.
  if (E->getType()->isExtVectorType()) {
    if (EltTy->isIntegerType() && Result.isInt())
      Result = APValue(HandleIntToIntCast(EltTy, SETy, Result.getInt(),
                                          Info.Ctx));
    else if (EltTy->isIntegerType())
      Result = APValue(HandleFloatToIntCast(EltTy, SETy, Result.getFloat(),
                                            Info.Ctx));
    else if (EltTy->isRealFloatingType() && Result.isInt())
      Result = APValue(HandleIntToFloatCast(EltTy, SETy, Result.getInt(),
                                            Info.Ctx));
    else if (EltTy->isRealFloatingType())
      Result = APValue(HandleFloatToFloatCast(EltTy, SETy, Result.getFloat(),
                                              Info.Ctx));
    else
      return APValue();

    // Splat and create vector APValue.
    llvm::SmallVector<APValue, 4> Elts(NElts, Result);
    return APValue(&Elts[0], Elts.size());
  }

  // For casts of a scalar to regular gcc-style vector type, bitcast the scalar
  // to the vector. To construct the APValue vector initializer, bitcast the
  // initializing value to an APInt, and shift out the bits pertaining to each
  // element.
  APSInt Init;
  Init = Result.isInt() ? Result.getInt() : Result.getFloat().bitcastToAPInt();

  llvm::SmallVector<APValue, 4> Elts;
  for (unsigned i = 0; i != NElts; ++i) {
    APSInt Tmp = Init;
    Tmp.extOrTrunc(EltWidth);

    if (EltTy->isIntegerType())
      Elts.push_back(APValue(Tmp));
    else if (EltTy->isRealFloatingType())
      Elts.push_back(APValue(APFloat(Tmp)));
    else
      return APValue();

    Init >>= EltWidth;
  }
  return APValue(&Elts[0], Elts.size());
}

APValue
VectorExprEvaluator::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  return this->Visit(const_cast<Expr*>(E->getInitializer()));
}

APValue
VectorExprEvaluator::VisitInitListExpr(const InitListExpr *E) {
  const VectorType *VT = E->getType()->getAs<VectorType>();
  unsigned NumInits = E->getNumInits();
  unsigned NumElements = VT->getNumElements();

  QualType EltTy = VT->getElementType();
  llvm::SmallVector<APValue, 4> Elements;

  // If a vector is initialized with a single element, that value
  // becomes every element of the vector, not just the first.
  // This is the behavior described in the IBM AltiVec documentation.
  if (NumInits == 1) {
    APValue InitValue;
    if (EltTy->isIntegerType()) {
      llvm::APSInt sInt(32);
      if (!EvaluateInteger(E->getInit(0), sInt, Info))
        return APValue();
      InitValue = APValue(sInt);
    } else {
      llvm::APFloat f(0.0);
      if (!EvaluateFloat(E->getInit(0), f, Info))
        return APValue();
      InitValue = APValue(f);
    }
    for (unsigned i = 0; i < NumElements; i++) {
      Elements.push_back(InitValue);
    }
  } else {
    for (unsigned i = 0; i < NumElements; i++) {
      if (EltTy->isIntegerType()) {
        llvm::APSInt sInt(32);
        if (i < NumInits) {
          if (!EvaluateInteger(E->getInit(i), sInt, Info))
            return APValue();
        } else {
          sInt = Info.Ctx.MakeIntValue(0, EltTy);
        }
        Elements.push_back(APValue(sInt));
      } else {
        llvm::APFloat f(0.0);
        if (i < NumInits) {
          if (!EvaluateFloat(E->getInit(i), f, Info))
            return APValue();
        } else {
          f = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy));
        }
        Elements.push_back(APValue(f));
      }
    }
  }
  return APValue(&Elements[0], Elements.size());
}

APValue
VectorExprEvaluator::GetZeroVector(QualType T) {
  const VectorType *VT = T->getAs<VectorType>();
  QualType EltTy = VT->getElementType();
  APValue ZeroElement;
  if (EltTy->isIntegerType())
    ZeroElement = APValue(Info.Ctx.MakeIntValue(0, EltTy));
  else
    ZeroElement =
        APValue(APFloat::getZero(Info.Ctx.getFloatTypeSemantics(EltTy)));

  llvm::SmallVector<APValue, 4> Elements(VT->getNumElements(), ZeroElement);
  return APValue(&Elements[0], Elements.size());
}

APValue VectorExprEvaluator::VisitConditionalOperator(const ConditionalOperator *E) {
  bool BoolResult;
  if (!HandleConversionToBool(E->getCond(), BoolResult, Info))
    return APValue();

  Expr* EvalExpr = BoolResult ? E->getTrueExpr() : E->getFalseExpr();

  APValue Result;
  if (EvaluateVector(EvalExpr, Result, Info))
    return Result;
  return APValue();
}

APValue VectorExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  return GetZeroVector(E->getType());
}

//===----------------------------------------------------------------------===//
// Integer Evaluation
//===----------------------------------------------------------------------===//

namespace {
class IntExprEvaluator
  : public StmtVisitor<IntExprEvaluator, bool> {
  EvalInfo &Info;
  APValue &Result;
public:
  IntExprEvaluator(EvalInfo &info, APValue &result)
    : Info(info), Result(result) {}

  bool Success(const llvm::APSInt &SI, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    assert(SI.isSigned() == E->getType()->isSignedIntegerType() &&
           "Invalid evaluation result.");
    assert(SI.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(SI);
    return true;
  }

  bool Success(const llvm::APInt &I, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    assert(I.getBitWidth() == Info.Ctx.getIntWidth(E->getType()) &&
           "Invalid evaluation result.");
    Result = APValue(APSInt(I));
    Result.getInt().setIsUnsigned(E->getType()->isUnsignedIntegerType());
    return true;
  }

  bool Success(uint64_t Value, const Expr *E) {
    assert(E->getType()->isIntegralOrEnumerationType() && 
           "Invalid evaluation result.");
    Result = APValue(Info.Ctx.MakeIntValue(Value, E->getType()));
    return true;
  }

  bool Error(SourceLocation L, diag::kind D, const Expr *E) {
    // Take the first error.
    if (Info.EvalResult.Diag == 0) {
      Info.EvalResult.DiagLoc = L;
      Info.EvalResult.Diag = D;
      Info.EvalResult.DiagExpr = E;
    }
    return false;
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitStmt(Stmt *) {
    assert(0 && "This should be called on integers, stmts are not integers");
    return false;
  }

  bool VisitExpr(Expr *E) {
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  bool VisitIntegerLiteral(const IntegerLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitCharacterLiteral(const CharacterLiteral *E) {
    return Success(E->getValue(), E);
  }
  bool VisitTypesCompatibleExpr(const TypesCompatibleExpr *E) {
    // Per gcc docs "this built-in function ignores top level
    // qualifiers".  We need to use the canonical version to properly
    // be able to strip CRV qualifiers from the type.
    QualType T0 = Info.Ctx.getCanonicalType(E->getArgType1());
    QualType T1 = Info.Ctx.getCanonicalType(E->getArgType2());
    return Success(Info.Ctx.typesAreCompatible(T0.getUnqualifiedType(),
                                               T1.getUnqualifiedType()),
                   E);
  }

  bool CheckReferencedDecl(const Expr *E, const Decl *D);
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    return CheckReferencedDecl(E, E->getDecl());
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (CheckReferencedDecl(E, E->getMemberDecl())) {
      // Conservatively assume a MemberExpr will have side-effects
      Info.EvalResult.HasSideEffects = true;
      return true;
    }
    return false;
  }

  bool VisitCallExpr(CallExpr *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitOffsetOfExpr(const OffsetOfExpr *E);
  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitConditionalOperator(const ConditionalOperator *E);

  bool VisitCastExpr(CastExpr* E);
  bool VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E);

  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return Success(E->getValue(), E);
  }

  bool VisitGNUNullExpr(const GNUNullExpr *E) {
    return Success(0, E);
  }

  bool VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    return Success(0, E);
  }

  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return Success(0, E);
  }

  bool VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return Success(E->EvaluateTrait(Info.Ctx), E);
  }

  bool VisitChooseExpr(const ChooseExpr *E) {
    return Visit(E->getChosenSubExpr(Info.Ctx));
  }

  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

private:
  CharUnits GetAlignOfExpr(const Expr *E);
  CharUnits GetAlignOfType(QualType T);
  static QualType GetObjectType(const Expr *E);
  bool TryEvaluateBuiltinObjectSize(CallExpr *E);
  // FIXME: Missing: array subscript of vector, member of vector
};
} // end anonymous namespace

static bool EvaluateIntegerOrLValue(const Expr* E, APValue &Result, EvalInfo &Info) {
  assert(E->getType()->isIntegralOrEnumerationType());
  return IntExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

static bool EvaluateInteger(const Expr* E, APSInt &Result, EvalInfo &Info) {
  assert(E->getType()->isIntegralOrEnumerationType());

  APValue Val;
  if (!EvaluateIntegerOrLValue(E, Val, Info) || !Val.isInt())
    return false;
  Result = Val.getInt();
  return true;
}

bool IntExprEvaluator::CheckReferencedDecl(const Expr* E, const Decl* D) {
  // Enums are integer constant exprs.
  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D))
    return Success(ECD->getInitVal(), E);

  // In C++, const, non-volatile integers initialized with ICEs are ICEs.
  // In C, they can also be folded, although they are not ICEs.
  if (Info.Ctx.getCanonicalType(E->getType()).getCVRQualifiers() 
                                                        == Qualifiers::Const) {

    if (isa<ParmVarDecl>(D))
      return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);

    if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (const Expr *Init = VD->getAnyInitializer()) {
        if (APValue *V = VD->getEvaluatedValue()) {
          if (V->isInt())
            return Success(V->getInt(), E);
          return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
        }

        if (VD->isEvaluatingValue())
          return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);

        VD->setEvaluatingValue();

        Expr::EvalResult EResult;
        if (Init->Evaluate(EResult, Info.Ctx) && !EResult.HasSideEffects &&
            EResult.Val.isInt()) {
          // Cache the evaluated value in the variable declaration.
          Result = EResult.Val;
          VD->setEvaluatedValue(Result);
          return true;
        }

        VD->setEvaluatedValue(APValue());
        return false;
      }
    }
  }

  // Otherwise, random variable references are not constants.
  return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
}

/// EvaluateBuiltinClassifyType - Evaluate __builtin_classify_type the same way
/// as GCC.
static int EvaluateBuiltinClassifyType(const CallExpr *E) {
  // The following enum mimics the values returned by GCC.
  // FIXME: Does GCC differ between lvalue and rvalue references here?
  enum gcc_type_class {
    no_type_class = -1,
    void_type_class, integer_type_class, char_type_class,
    enumeral_type_class, boolean_type_class,
    pointer_type_class, reference_type_class, offset_type_class,
    real_type_class, complex_type_class,
    function_type_class, method_type_class,
    record_type_class, union_type_class,
    array_type_class, string_type_class,
    lang_type_class
  };

  // If no argument was supplied, default to "no_type_class". This isn't
  // ideal, however it is what gcc does.
  if (E->getNumArgs() == 0)
    return no_type_class;

  QualType ArgTy = E->getArg(0)->getType();
  if (ArgTy->isVoidType())
    return void_type_class;
  else if (ArgTy->isEnumeralType())
    return enumeral_type_class;
  else if (ArgTy->isBooleanType())
    return boolean_type_class;
  else if (ArgTy->isCharType())
    return string_type_class; // gcc doesn't appear to use char_type_class
  else if (ArgTy->isIntegerType())
    return integer_type_class;
  else if (ArgTy->isPointerType())
    return pointer_type_class;
  else if (ArgTy->isReferenceType())
    return reference_type_class;
  else if (ArgTy->isRealType())
    return real_type_class;
  else if (ArgTy->isComplexType())
    return complex_type_class;
  else if (ArgTy->isFunctionType())
    return function_type_class;
  else if (ArgTy->isStructureOrClassType())
    return record_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else if (ArgTy->isArrayType())
    return array_type_class;
  else if (ArgTy->isUnionType())
    return union_type_class;
  else  // FIXME: offset_type_class, method_type_class, & lang_type_class?
    assert(0 && "CallExpr::isBuiltinClassifyType(): unimplemented type");
  return -1;
}

/// Retrieves the "underlying object type" of the given expression,
/// as used by __builtin_object_size.
QualType IntExprEvaluator::GetObjectType(const Expr *E) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return VD->getType();
  } else if (isa<CompoundLiteralExpr>(E)) {
    return E->getType();
  }

  return QualType();
}

bool IntExprEvaluator::TryEvaluateBuiltinObjectSize(CallExpr *E) {
  // TODO: Perhaps we should let LLVM lower this?
  LValue Base;
  if (!EvaluatePointer(E->getArg(0), Base, Info))
    return false;

  // If we can prove the base is null, lower to zero now.
  const Expr *LVBase = Base.getLValueBase();
  if (!LVBase) return Success(0, E);

  QualType T = GetObjectType(LVBase);
  if (T.isNull() ||
      T->isIncompleteType() ||
      T->isFunctionType() ||
      T->isVariablyModifiedType() ||
      T->isDependentType())
    return false;

  CharUnits Size = Info.Ctx.getTypeSizeInChars(T);
  CharUnits Offset = Base.getLValueOffset();

  if (!Offset.isNegative() && Offset <= Size)
    Size -= Offset;
  else
    Size = CharUnits::Zero();
  return Success(Size.getQuantity(), E);
}

bool IntExprEvaluator::VisitCallExpr(CallExpr *E) {
  switch (E->isBuiltinCall(Info.Ctx)) {
  default:
    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);

  case Builtin::BI__builtin_object_size: {
    if (TryEvaluateBuiltinObjectSize(E))
      return true;

    // If evaluating the argument has side-effects we can't determine
    // the size of the object and lower it to unknown now.
    if (E->getArg(0)->HasSideEffects(Info.Ctx)) {
      if (E->getArg(1)->EvaluateAsInt(Info.Ctx).getZExtValue() <= 1)
        return Success(-1ULL, E);
      return Success(0, E);
    }

    return Error(E->getLocStart(), diag::note_invalid_subexpr_in_ice, E);
  }

  case Builtin::BI__builtin_classify_type:
    return Success(EvaluateBuiltinClassifyType(E), E);

  case Builtin::BI__builtin_constant_p:
    // __builtin_constant_p always has one operand: it returns true if that
    // operand can be folded, false otherwise.
    return Success(E->getArg(0)->isEvaluatable(Info.Ctx), E);
      
  case Builtin::BI__builtin_eh_return_data_regno: {
    int Operand = E->getArg(0)->EvaluateAsInt(Info.Ctx).getZExtValue();
    Operand = Info.Ctx.Target.getEHDataRegisterNumber(Operand);
    return Success(Operand, E);
  }

  case Builtin::BI__builtin_expect:
    return Visit(E->getArg(0));
  }
}

bool IntExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_Comma) {
    if (!Visit(E->getRHS()))
      return false;

    // If we can't evaluate the LHS, it might have side effects;
    // conservatively mark it.
    if (!E->getLHS()->isEvaluatable(Info.Ctx))
      Info.EvalResult.HasSideEffects = true;

    return true;
  }

  if (E->isLogicalOp()) {
    // These need to be handled specially because the operands aren't
    // necessarily integral
    bool lhsResult, rhsResult;

    if (HandleConversionToBool(E->getLHS(), lhsResult, Info)) {
      // We were able to evaluate the LHS, see if we can get away with not
      // evaluating the RHS: 0 && X -> 0, 1 || X -> 1
      if (lhsResult == (E->getOpcode() == BO_LOr))
        return Success(lhsResult, E);

      if (HandleConversionToBool(E->getRHS(), rhsResult, Info)) {
        if (E->getOpcode() == BO_LOr)
          return Success(lhsResult || rhsResult, E);
        else
          return Success(lhsResult && rhsResult, E);
      }
    } else {
      if (HandleConversionToBool(E->getRHS(), rhsResult, Info)) {
        // We can't evaluate the LHS; however, sometimes the result
        // is determined by the RHS: X && 0 -> 0, X || 1 -> 1.
        if (rhsResult == (E->getOpcode() == BO_LOr) ||
            !rhsResult == (E->getOpcode() == BO_LAnd)) {
          // Since we weren't able to evaluate the left hand side, it
          // must have had side effects.
          Info.EvalResult.HasSideEffects = true;

          return Success(rhsResult, E);
        }
      }
    }

    return false;
  }

  QualType LHSTy = E->getLHS()->getType();
  QualType RHSTy = E->getRHS()->getType();

  if (LHSTy->isAnyComplexType()) {
    assert(RHSTy->isAnyComplexType() && "Invalid comparison");
    ComplexValue LHS, RHS;

    if (!EvaluateComplex(E->getLHS(), LHS, Info))
      return false;

    if (!EvaluateComplex(E->getRHS(), RHS, Info))
      return false;

    if (LHS.isComplexFloat()) {
      APFloat::cmpResult CR_r =
        LHS.getComplexFloatReal().compare(RHS.getComplexFloatReal());
      APFloat::cmpResult CR_i =
        LHS.getComplexFloatImag().compare(RHS.getComplexFloatImag());

      if (E->getOpcode() == BO_EQ)
        return Success((CR_r == APFloat::cmpEqual &&
                        CR_i == APFloat::cmpEqual), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid complex comparison.");
        return Success(((CR_r == APFloat::cmpGreaterThan ||
                         CR_r == APFloat::cmpLessThan ||
                         CR_r == APFloat::cmpUnordered) ||
                        (CR_i == APFloat::cmpGreaterThan ||
                         CR_i == APFloat::cmpLessThan ||
                         CR_i == APFloat::cmpUnordered)), E);
      }
    } else {
      if (E->getOpcode() == BO_EQ)
        return Success((LHS.getComplexIntReal() == RHS.getComplexIntReal() &&
                        LHS.getComplexIntImag() == RHS.getComplexIntImag()), E);
      else {
        assert(E->getOpcode() == BO_NE &&
               "Invalid compex comparison.");
        return Success((LHS.getComplexIntReal() != RHS.getComplexIntReal() ||
                        LHS.getComplexIntImag() != RHS.getComplexIntImag()), E);
      }
    }
  }

  if (LHSTy->isRealFloatingType() &&
      RHSTy->isRealFloatingType()) {
    APFloat RHS(0.0), LHS(0.0);

    if (!EvaluateFloat(E->getRHS(), RHS, Info))
      return false;

    if (!EvaluateFloat(E->getLHS(), LHS, Info))
      return false;

    APFloat::cmpResult CR = LHS.compare(RHS);

    switch (E->getOpcode()) {
    default:
      assert(0 && "Invalid binary operator!");
    case BO_LT:
      return Success(CR == APFloat::cmpLessThan, E);
    case BO_GT:
      return Success(CR == APFloat::cmpGreaterThan, E);
    case BO_LE:
      return Success(CR == APFloat::cmpLessThan || CR == APFloat::cmpEqual, E);
    case BO_GE:
      return Success(CR == APFloat::cmpGreaterThan || CR == APFloat::cmpEqual,
                     E);
    case BO_EQ:
      return Success(CR == APFloat::cmpEqual, E);
    case BO_NE:
      return Success(CR == APFloat::cmpGreaterThan
                     || CR == APFloat::cmpLessThan
                     || CR == APFloat::cmpUnordered, E);
    }
  }

  if (LHSTy->isPointerType() && RHSTy->isPointerType()) {
    if (E->getOpcode() == BO_Sub || E->isEqualityOp()) {
      LValue LHSValue;
      if (!EvaluatePointer(E->getLHS(), LHSValue, Info))
        return false;

      LValue RHSValue;
      if (!EvaluatePointer(E->getRHS(), RHSValue, Info))
        return false;

      // Reject any bases from the normal codepath; we special-case comparisons
      // to null.
      if (LHSValue.getLValueBase()) {
        if (!E->isEqualityOp())
          return false;
        if (RHSValue.getLValueBase() || !RHSValue.getLValueOffset().isZero())
          return false;
        bool bres;
        if (!EvalPointerValueAsBool(LHSValue, bres))
          return false;
        return Success(bres ^ (E->getOpcode() == BO_EQ), E);
      } else if (RHSValue.getLValueBase()) {
        if (!E->isEqualityOp())
          return false;
        if (LHSValue.getLValueBase() || !LHSValue.getLValueOffset().isZero())
          return false;
        bool bres;
        if (!EvalPointerValueAsBool(RHSValue, bres))
          return false;
        return Success(bres ^ (E->getOpcode() == BO_EQ), E);
      }

      if (E->getOpcode() == BO_Sub) {
        QualType Type = E->getLHS()->getType();
        QualType ElementType = Type->getAs<PointerType>()->getPointeeType();

        CharUnits ElementSize = CharUnits::One();
        if (!ElementType->isVoidType() && !ElementType->isFunctionType())
          ElementSize = Info.Ctx.getTypeSizeInChars(ElementType);

        CharUnits Diff = LHSValue.getLValueOffset() - 
                             RHSValue.getLValueOffset();
        return Success(Diff / ElementSize, E);
      }
      bool Result;
      if (E->getOpcode() == BO_EQ) {
        Result = LHSValue.getLValueOffset() == RHSValue.getLValueOffset();
      } else {
        Result = LHSValue.getLValueOffset() != RHSValue.getLValueOffset();
      }
      return Success(Result, E);
    }
  }
  if (!LHSTy->isIntegralOrEnumerationType() ||
      !RHSTy->isIntegralOrEnumerationType()) {
    // We can't continue from here for non-integral types, and they
    // could potentially confuse the following operations.
    return false;
  }

  // The LHS of a constant expr is always evaluated and needed.
  if (!Visit(E->getLHS()))
    return false; // error in subexpression.

  APValue RHSVal;
  if (!EvaluateIntegerOrLValue(E->getRHS(), RHSVal, Info))
    return false;

  // Handle cases like (unsigned long)&a + 4.
  if (E->isAdditiveOp() && Result.isLValue() && RHSVal.isInt()) {
    CharUnits Offset = Result.getLValueOffset();
    CharUnits AdditionalOffset = CharUnits::fromQuantity(
                                     RHSVal.getInt().getZExtValue());
    if (E->getOpcode() == BO_Add)
      Offset += AdditionalOffset;
    else
      Offset -= AdditionalOffset;
    Result = APValue(Result.getLValueBase(), Offset);
    return true;
  }

  // Handle cases like 4 + (unsigned long)&a
  if (E->getOpcode() == BO_Add &&
        RHSVal.isLValue() && Result.isInt()) {
    CharUnits Offset = RHSVal.getLValueOffset();
    Offset += CharUnits::fromQuantity(Result.getInt().getZExtValue());
    Result = APValue(RHSVal.getLValueBase(), Offset);
    return true;
  }

  // All the following cases expect both operands to be an integer
  if (!Result.isInt() || !RHSVal.isInt())
    return false;

  APSInt& RHS = RHSVal.getInt();

  switch (E->getOpcode()) {
  default:
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case BO_Mul: return Success(Result.getInt() * RHS, E);
  case BO_Add: return Success(Result.getInt() + RHS, E);
  case BO_Sub: return Success(Result.getInt() - RHS, E);
  case BO_And: return Success(Result.getInt() & RHS, E);
  case BO_Xor: return Success(Result.getInt() ^ RHS, E);
  case BO_Or:  return Success(Result.getInt() | RHS, E);
  case BO_Div:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(Result.getInt() / RHS, E);
  case BO_Rem:
    if (RHS == 0)
      return Error(E->getOperatorLoc(), diag::note_expr_divide_by_zero, E);
    return Success(Result.getInt() % RHS, E);
  case BO_Shl: {
    // FIXME: Warn about out of range shift amounts!
    unsigned SA =
      (unsigned) RHS.getLimitedValue(Result.getInt().getBitWidth()-1);
    return Success(Result.getInt() << SA, E);
  }
  case BO_Shr: {
    unsigned SA =
      (unsigned) RHS.getLimitedValue(Result.getInt().getBitWidth()-1);
    return Success(Result.getInt() >> SA, E);
  }

  case BO_LT: return Success(Result.getInt() < RHS, E);
  case BO_GT: return Success(Result.getInt() > RHS, E);
  case BO_LE: return Success(Result.getInt() <= RHS, E);
  case BO_GE: return Success(Result.getInt() >= RHS, E);
  case BO_EQ: return Success(Result.getInt() == RHS, E);
  case BO_NE: return Success(Result.getInt() != RHS, E);
  }
}

bool IntExprEvaluator::VisitConditionalOperator(const ConditionalOperator *E) {
  bool Cond;
  if (!HandleConversionToBool(E->getCond(), Cond, Info))
    return false;

  return Visit(Cond ? E->getTrueExpr() : E->getFalseExpr());
}

CharUnits IntExprEvaluator::GetAlignOfType(QualType T) {
  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = T->getAs<ReferenceType>())
    T = Ref->getPointeeType();

  // Get information about the alignment.
  unsigned CharSize = Info.Ctx.Target.getCharWidth();

  // __alignof is defined to return the preferred alignment.
  return CharUnits::fromQuantity(
      Info.Ctx.getPreferredTypeAlign(T.getTypePtr()) / CharSize);
}

CharUnits IntExprEvaluator::GetAlignOfExpr(const Expr *E) {
  E = E->IgnoreParens();

  // alignof decl is always accepted, even if it doesn't make sense: we default
  // to 1 in those cases.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return Info.Ctx.getDeclAlign(DRE->getDecl(), 
                                 /*RefAsPointee*/true);

  if (const MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return Info.Ctx.getDeclAlign(ME->getMemberDecl(),
                                 /*RefAsPointee*/true);

  return GetAlignOfType(E->getType());
}


/// VisitSizeAlignOfExpr - Evaluate a sizeof or alignof with a result as the
/// expression's type.
bool IntExprEvaluator::VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E) {
  // Handle alignof separately.
  if (!E->isSizeOf()) {
    if (E->isArgumentType())
      return Success(GetAlignOfType(E->getArgumentType()).getQuantity(), E);
    else
      return Success(GetAlignOfExpr(E->getArgumentExpr()).getQuantity(), E);
  }

  QualType SrcTy = E->getTypeOfArgument();
  // C++ [expr.sizeof]p2: "When applied to a reference or a reference type,
  //   the result is the size of the referenced type."
  // C++ [expr.alignof]p3: "When alignof is applied to a reference type, the
  //   result shall be the alignment of the referenced type."
  if (const ReferenceType *Ref = SrcTy->getAs<ReferenceType>())
    SrcTy = Ref->getPointeeType();

  // sizeof(void), __alignof__(void), sizeof(function) = 1 as a gcc
  // extension.
  if (SrcTy->isVoidType() || SrcTy->isFunctionType())
    return Success(1, E);

  // sizeof(vla) is not a constantexpr: C99 6.5.3.4p2.
  if (!SrcTy->isConstantSizeType())
    return false;

  // Get information about the size.
  return Success(Info.Ctx.getTypeSizeInChars(SrcTy).getQuantity(), E);
}

bool IntExprEvaluator::VisitOffsetOfExpr(const OffsetOfExpr *E) {
  CharUnits Result;
  unsigned n = E->getNumComponents();
  OffsetOfExpr* OOE = const_cast<OffsetOfExpr*>(E);
  if (n == 0)
    return false;
  QualType CurrentType = E->getTypeSourceInfo()->getType();
  for (unsigned i = 0; i != n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = OOE->getComponent(i);
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array: {
      Expr *Idx = OOE->getIndexExpr(ON.getArrayExprIndex());
      APSInt IdxResult;
      if (!EvaluateInteger(Idx, IdxResult, Info))
        return false;
      const ArrayType *AT = Info.Ctx.getAsArrayType(CurrentType);
      if (!AT)
        return false;
      CurrentType = AT->getElementType();
      CharUnits ElementSize = Info.Ctx.getTypeSizeInChars(CurrentType);
      Result += IdxResult.getSExtValue() * ElementSize;
        break;
    }
        
    case OffsetOfExpr::OffsetOfNode::Field: {
      FieldDecl *MemberDecl = ON.getField();
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT) 
        return false;
      RecordDecl *RD = RT->getDecl();
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);
      unsigned i = 0;
      // FIXME: It would be nice if we didn't have to loop here!
      for (RecordDecl::field_iterator Field = RD->field_begin(),
                                      FieldEnd = RD->field_end();
           Field != FieldEnd; (void)++Field, ++i) {
        if (*Field == MemberDecl)
          break;
      }
      assert(i < RL.getFieldCount() && "offsetof field in wrong type");
      Result += CharUnits::fromQuantity(
                           RL.getFieldOffset(i) / Info.Ctx.getCharWidth());
      CurrentType = MemberDecl->getType().getNonReferenceType();
      break;
    }
        
    case OffsetOfExpr::OffsetOfNode::Identifier:
      llvm_unreachable("dependent __builtin_offsetof");
      return false;
        
    case OffsetOfExpr::OffsetOfNode::Base: {
      CXXBaseSpecifier *BaseSpec = ON.getBase();
      if (BaseSpec->isVirtual())
        return false;

      // Find the layout of the class whose base we are looking into.
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT) 
        return false;
      RecordDecl *RD = RT->getDecl();
      const ASTRecordLayout &RL = Info.Ctx.getASTRecordLayout(RD);

      // Find the base class itself.
      CurrentType = BaseSpec->getType();
      const RecordType *BaseRT = CurrentType->getAs<RecordType>();
      if (!BaseRT)
        return false;
      
      // Add the offset to the base.
      Result += CharUnits::fromQuantity(
                RL.getBaseClassOffset(cast<CXXRecordDecl>(BaseRT->getDecl()))
                                        / Info.Ctx.getCharWidth());
      break;
    }
    }
  }
  return Success(Result.getQuantity(), E);
}

bool IntExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->getOpcode() == UO_LNot) {
    // LNot's operand isn't necessarily an integer, so we handle it specially.
    bool bres;
    if (!HandleConversionToBool(E->getSubExpr(), bres, Info))
      return false;
    return Success(!bres, E);
  }

  // Only handle integral operations...
  if (!E->getSubExpr()->getType()->isIntegralOrEnumerationType())
    return false;

  // Get the operand value into 'Result'.
  if (!Visit(E->getSubExpr()))
    return false;

  switch (E->getOpcode()) {
  default:
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    return Error(E->getOperatorLoc(), diag::note_invalid_subexpr_in_ice, E);
  case UO_Extension:
    // FIXME: Should extension allow i-c-e extension expressions in its scope?
    // If so, we could clear the diagnostic ID.
    return true;
  case UO_Plus:
    // The result is always just the subexpr.
    return true;
  case UO_Minus:
    if (!Result.isInt()) return false;
    return Success(-Result.getInt(), E);
  case UO_Not:
    if (!Result.isInt()) return false;
    return Success(~Result.getInt(), E);
  }
}

/// HandleCast - This is used to evaluate implicit or explicit casts where the
/// result type is integer.
bool IntExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr *SubExpr = E->getSubExpr();
  QualType DestType = E->getType();
  QualType SrcType = SubExpr->getType();

  if (DestType->isBooleanType()) {
    bool BoolResult;
    if (!HandleConversionToBool(SubExpr, BoolResult, Info))
      return false;
    return Success(BoolResult, E);
  }

  // Handle simple integer->integer casts.
  if (SrcType->isIntegralOrEnumerationType()) {
    if (!Visit(SubExpr))
      return false;

    if (!Result.isInt()) {
      // Only allow casts of lvalues if they are lossless.
      return Info.Ctx.getTypeSize(DestType) == Info.Ctx.getTypeSize(SrcType);
    }

    return Success(HandleIntToIntCast(DestType, SrcType,
                                      Result.getInt(), Info.Ctx), E);
  }

  // FIXME: Clean this up!
  if (SrcType->isPointerType()) {
    LValue LV;
    if (!EvaluatePointer(SubExpr, LV, Info))
      return false;

    if (LV.getLValueBase()) {
      // Only allow based lvalue casts if they are lossless.
      if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(SrcType))
        return false;

      LV.moveInto(Result);
      return true;
    }

    APSInt AsInt = Info.Ctx.MakeIntValue(LV.getLValueOffset().getQuantity(), 
                                         SrcType);
    return Success(HandleIntToIntCast(DestType, SrcType, AsInt, Info.Ctx), E);
  }

  if (SrcType->isArrayType() || SrcType->isFunctionType()) {
    // This handles double-conversion cases, where there's both
    // an l-value promotion and an implicit conversion to int.
    LValue LV;
    if (!EvaluateLValue(SubExpr, LV, Info))
      return false;

    if (Info.Ctx.getTypeSize(DestType) != Info.Ctx.getTypeSize(Info.Ctx.VoidPtrTy))
      return false;

    LV.moveInto(Result);
    return true;
  }

  if (SrcType->isAnyComplexType()) {
    ComplexValue C;
    if (!EvaluateComplex(SubExpr, C, Info))
      return false;
    if (C.isComplexFloat())
      return Success(HandleFloatToIntCast(DestType, SrcType,
                                          C.getComplexFloatReal(), Info.Ctx),
                     E);
    else
      return Success(HandleIntToIntCast(DestType, SrcType,
                                        C.getComplexIntReal(), Info.Ctx), E);
  }
  // FIXME: Handle vectors

  if (!SrcType->isRealFloatingType())
    return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);

  APFloat F(0.0);
  if (!EvaluateFloat(SubExpr, F, Info))
    return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);

  return Success(HandleFloatToIntCast(DestType, SrcType, F, Info.Ctx), E);
}

bool IntExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntReal(), E);
  }

  return Visit(E->getSubExpr());
}

bool IntExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isComplexIntegerType()) {
    ComplexValue LV;
    if (!EvaluateComplex(E->getSubExpr(), LV, Info) || !LV.isComplexInt())
      return Error(E->getExprLoc(), diag::note_invalid_subexpr_in_ice, E);
    return Success(LV.getComplexIntImag(), E);
  }

  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  return Success(0, E);
}

//===----------------------------------------------------------------------===//
// Float Evaluation
//===----------------------------------------------------------------------===//

namespace {
class FloatExprEvaluator
  : public StmtVisitor<FloatExprEvaluator, bool> {
  EvalInfo &Info;
  APFloat &Result;
public:
  FloatExprEvaluator(EvalInfo &info, APFloat &result)
    : Info(info), Result(result) {}

  bool VisitStmt(Stmt *S) {
    return false;
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }
  bool VisitCallExpr(const CallExpr *E);

  bool VisitUnaryOperator(const UnaryOperator *E);
  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitFloatingLiteral(const FloatingLiteral *E);
  bool VisitCastExpr(CastExpr *E);
  bool VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
  bool VisitConditionalOperator(ConditionalOperator *E);

  bool VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitUnaryExtension(const UnaryOperator *E)
    { return Visit(E->getSubExpr()); }
  bool VisitUnaryReal(const UnaryOperator *E);
  bool VisitUnaryImag(const UnaryOperator *E);

  // FIXME: Missing: array subscript of vector, member of vector,
  //                 ImplicitValueInitExpr
};
} // end anonymous namespace

static bool EvaluateFloat(const Expr* E, APFloat& Result, EvalInfo &Info) {
  assert(E->getType()->isRealFloatingType());
  return FloatExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

static bool TryEvaluateBuiltinNaN(ASTContext &Context,
                                  QualType ResultTy,
                                  const Expr *Arg,
                                  bool SNaN,
                                  llvm::APFloat &Result) {
  const StringLiteral *S = dyn_cast<StringLiteral>(Arg->IgnoreParenCasts());
  if (!S) return false;

  const llvm::fltSemantics &Sem = Context.getFloatTypeSemantics(ResultTy);

  llvm::APInt fill;

  // Treat empty strings as if they were zero.
  if (S->getString().empty())
    fill = llvm::APInt(32, 0);
  else if (S->getString().getAsInteger(0, fill))
    return false;

  if (SNaN)
    Result = llvm::APFloat::getSNaN(Sem, false, &fill);
  else
    Result = llvm::APFloat::getQNaN(Sem, false, &fill);
  return true;
}

bool FloatExprEvaluator::VisitCallExpr(const CallExpr *E) {
  switch (E->isBuiltinCall(Info.Ctx)) {
  default: return false;
  case Builtin::BI__builtin_huge_val:
  case Builtin::BI__builtin_huge_valf:
  case Builtin::BI__builtin_huge_vall:
  case Builtin::BI__builtin_inf:
  case Builtin::BI__builtin_inff:
  case Builtin::BI__builtin_infl: {
    const llvm::fltSemantics &Sem =
      Info.Ctx.getFloatTypeSemantics(E->getType());
    Result = llvm::APFloat::getInf(Sem);
    return true;
  }

  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
    return TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                                 true, Result);

  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
    // If this is __builtin_nan() turn this into a nan, otherwise we
    // can't constant fold it.
    return TryEvaluateBuiltinNaN(Info.Ctx, E->getType(), E->getArg(0),
                                 false, Result);

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
    if (!EvaluateFloat(E->getArg(0), Result, Info))
      return false;

    if (Result.isNegative())
      Result.changeSign();
    return true;

  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl: {
    APFloat RHS(0.);
    if (!EvaluateFloat(E->getArg(0), Result, Info) ||
        !EvaluateFloat(E->getArg(1), RHS, Info))
      return false;
    Result.copySign(RHS);
    return true;
  }
  }
}

bool FloatExprEvaluator::VisitUnaryReal(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatReal;
    return true;
  }

  return Visit(E->getSubExpr());
}

bool FloatExprEvaluator::VisitUnaryImag(const UnaryOperator *E) {
  if (E->getSubExpr()->getType()->isAnyComplexType()) {
    ComplexValue CV;
    if (!EvaluateComplex(E->getSubExpr(), CV, Info))
      return false;
    Result = CV.FloatImag;
    return true;
  }

  if (!E->getSubExpr()->isEvaluatable(Info.Ctx))
    Info.EvalResult.HasSideEffects = true;
  const llvm::fltSemantics &Sem = Info.Ctx.getFloatTypeSemantics(E->getType());
  Result = llvm::APFloat::getZero(Sem);
  return true;
}

bool FloatExprEvaluator::VisitUnaryOperator(const UnaryOperator *E) {
  if (E->getOpcode() == UO_Deref)
    return false;

  if (!EvaluateFloat(E->getSubExpr(), Result, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case UO_Plus:
    return true;
  case UO_Minus:
    Result.changeSign();
    return true;
  }
}

bool FloatExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_Comma) {
    if (!EvaluateFloat(E->getRHS(), Result, Info))
      return false;

    // If we can't evaluate the LHS, it might have side effects;
    // conservatively mark it.
    if (!E->getLHS()->isEvaluatable(Info.Ctx))
      Info.EvalResult.HasSideEffects = true;

    return true;
  }

  // FIXME: Diagnostics?  I really don't understand how the warnings
  // and errors are supposed to work.
  APFloat RHS(0.0);
  if (!EvaluateFloat(E->getLHS(), Result, Info))
    return false;
  if (!EvaluateFloat(E->getRHS(), RHS, Info))
    return false;

  switch (E->getOpcode()) {
  default: return false;
  case BO_Mul:
    Result.multiply(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Add:
    Result.add(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Sub:
    Result.subtract(RHS, APFloat::rmNearestTiesToEven);
    return true;
  case BO_Div:
    Result.divide(RHS, APFloat::rmNearestTiesToEven);
    return true;
  }
}

bool FloatExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *E) {
  Result = E->getValue();
  return true;
}

bool FloatExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr* SubExpr = E->getSubExpr();

  if (SubExpr->getType()->isIntegralOrEnumerationType()) {
    APSInt IntResult;
    if (!EvaluateInteger(SubExpr, IntResult, Info))
      return false;
    Result = HandleIntToFloatCast(E->getType(), SubExpr->getType(),
                                  IntResult, Info.Ctx);
    return true;
  }
  if (SubExpr->getType()->isRealFloatingType()) {
    if (!Visit(SubExpr))
      return false;
    Result = HandleFloatToFloatCast(E->getType(), SubExpr->getType(),
                                    Result, Info.Ctx);
    return true;
  }
  // FIXME: Handle complex types

  return false;
}

bool FloatExprEvaluator::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  Result = APFloat::getZero(Info.Ctx.getFloatTypeSemantics(E->getType()));
  return true;
}

bool FloatExprEvaluator::VisitConditionalOperator(ConditionalOperator *E) {
  bool Cond;
  if (!HandleConversionToBool(E->getCond(), Cond, Info))
    return false;

  return Visit(Cond ? E->getTrueExpr() : E->getFalseExpr());
}

//===----------------------------------------------------------------------===//
// Complex Evaluation (for float and integer)
//===----------------------------------------------------------------------===//

namespace {
class ComplexExprEvaluator
  : public StmtVisitor<ComplexExprEvaluator, bool> {
  EvalInfo &Info;
  ComplexValue &Result;

public:
  ComplexExprEvaluator(EvalInfo &info, ComplexValue &Result)
    : Info(info), Result(Result) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  bool VisitStmt(Stmt *S) {
    return false;
  }

  bool VisitParenExpr(ParenExpr *E) { return Visit(E->getSubExpr()); }

  bool VisitImaginaryLiteral(ImaginaryLiteral *E);

  bool VisitCastExpr(CastExpr *E);

  bool VisitBinaryOperator(const BinaryOperator *E);
  bool VisitChooseExpr(const ChooseExpr *E)
    { return Visit(E->getChosenSubExpr(Info.Ctx)); }
  bool VisitUnaryExtension(const UnaryOperator *E)
    { return Visit(E->getSubExpr()); }
  // FIXME Missing: unary +/-/~, binary div, ImplicitValueInitExpr,
  //                conditional ?:, comma
};
} // end anonymous namespace

static bool EvaluateComplex(const Expr *E, ComplexValue &Result,
                            EvalInfo &Info) {
  assert(E->getType()->isAnyComplexType());
  return ComplexExprEvaluator(Info, Result).Visit(const_cast<Expr*>(E));
}

bool ComplexExprEvaluator::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  Expr* SubExpr = E->getSubExpr();

  if (SubExpr->getType()->isRealFloatingType()) {
    Result.makeComplexFloat();
    APFloat &Imag = Result.FloatImag;
    if (!EvaluateFloat(SubExpr, Imag, Info))
      return false;

    Result.FloatReal = APFloat(Imag.getSemantics());
    return true;
  } else {
    assert(SubExpr->getType()->isIntegerType() &&
           "Unexpected imaginary literal.");

    Result.makeComplexInt();
    APSInt &Imag = Result.IntImag;
    if (!EvaluateInteger(SubExpr, Imag, Info))
      return false;

    Result.IntReal = APSInt(Imag.getBitWidth(), !Imag.isSigned());
    return true;
  }
}

bool ComplexExprEvaluator::VisitCastExpr(CastExpr *E) {
  Expr* SubExpr = E->getSubExpr();
  QualType EltType = E->getType()->getAs<ComplexType>()->getElementType();
  QualType SubType = SubExpr->getType();

  if (SubType->isRealFloatingType()) {
    APFloat &Real = Result.FloatReal;
    if (!EvaluateFloat(SubExpr, Real, Info))
      return false;

    if (EltType->isRealFloatingType()) {
      Result.makeComplexFloat();
      Real = HandleFloatToFloatCast(EltType, SubType, Real, Info.Ctx);
      Result.FloatImag = APFloat(Real.getSemantics());
      return true;
    } else {
      Result.makeComplexInt();
      Result.IntReal = HandleFloatToIntCast(EltType, SubType, Real, Info.Ctx);
      Result.IntImag = APSInt(Result.IntReal.getBitWidth(),
                              !Result.IntReal.isSigned());
      return true;
    }
  } else if (SubType->isIntegerType()) {
    APSInt &Real = Result.IntReal;
    if (!EvaluateInteger(SubExpr, Real, Info))
      return false;

    if (EltType->isRealFloatingType()) {
      Result.makeComplexFloat();
      Result.FloatReal
        = HandleIntToFloatCast(EltType, SubType, Real, Info.Ctx);
      Result.FloatImag = APFloat(Result.FloatReal.getSemantics());
      return true;
    } else {
      Result.makeComplexInt();
      Real = HandleIntToIntCast(EltType, SubType, Real, Info.Ctx);
      Result.IntImag = APSInt(Real.getBitWidth(), !Real.isSigned());
      return true;
    }
  } else if (const ComplexType *CT = SubType->getAs<ComplexType>()) {
    if (!Visit(SubExpr))
      return false;

    QualType SrcType = CT->getElementType();

    if (Result.isComplexFloat()) {
      if (EltType->isRealFloatingType()) {
        Result.makeComplexFloat();
        Result.FloatReal = HandleFloatToFloatCast(EltType, SrcType,
                                                  Result.FloatReal,
                                                  Info.Ctx);
        Result.FloatImag = HandleFloatToFloatCast(EltType, SrcType,
                                                  Result.FloatImag,
                                                  Info.Ctx);
        return true;
      } else {
        Result.makeComplexInt();
        Result.IntReal = HandleFloatToIntCast(EltType, SrcType,
                                              Result.FloatReal,
                                              Info.Ctx);
        Result.IntImag = HandleFloatToIntCast(EltType, SrcType,
                                              Result.FloatImag,
                                              Info.Ctx);
        return true;
      }
    } else {
      assert(Result.isComplexInt() && "Invalid evaluate result.");
      if (EltType->isRealFloatingType()) {
        Result.makeComplexFloat();
        Result.FloatReal = HandleIntToFloatCast(EltType, SrcType,
                                                Result.IntReal,
                                                Info.Ctx);
        Result.FloatImag = HandleIntToFloatCast(EltType, SrcType,
                                                Result.IntImag,
                                                Info.Ctx);
        return true;
      } else {
        Result.makeComplexInt();
        Result.IntReal = HandleIntToIntCast(EltType, SrcType,
                                            Result.IntReal,
                                            Info.Ctx);
        Result.IntImag = HandleIntToIntCast(EltType, SrcType,
                                            Result.IntImag,
                                            Info.Ctx);
        return true;
      }
    }
  }

  // FIXME: Handle more casts.
  return false;
}

bool ComplexExprEvaluator::VisitBinaryOperator(const BinaryOperator *E) {
  if (!Visit(E->getLHS()))
    return false;

  ComplexValue RHS;
  if (!EvaluateComplex(E->getRHS(), RHS, Info))
    return false;

  assert(Result.isComplexFloat() == RHS.isComplexFloat() &&
         "Invalid operands to binary operator.");
  switch (E->getOpcode()) {
  default: return false;
  case BO_Add:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().add(RHS.getComplexFloatReal(),
                                       APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(RHS.getComplexFloatImag(),
                                       APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() += RHS.getComplexIntReal();
      Result.getComplexIntImag() += RHS.getComplexIntImag();
    }
    break;
  case BO_Sub:
    if (Result.isComplexFloat()) {
      Result.getComplexFloatReal().subtract(RHS.getComplexFloatReal(),
                                            APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().subtract(RHS.getComplexFloatImag(),
                                            APFloat::rmNearestTiesToEven);
    } else {
      Result.getComplexIntReal() -= RHS.getComplexIntReal();
      Result.getComplexIntImag() -= RHS.getComplexIntImag();
    }
    break;
  case BO_Mul:
    if (Result.isComplexFloat()) {
      ComplexValue LHS = Result;
      APFloat &LHS_r = LHS.getComplexFloatReal();
      APFloat &LHS_i = LHS.getComplexFloatImag();
      APFloat &RHS_r = RHS.getComplexFloatReal();
      APFloat &RHS_i = RHS.getComplexFloatImag();

      APFloat Tmp = LHS_r;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatReal().subtract(Tmp, APFloat::rmNearestTiesToEven);

      Tmp = LHS_r;
      Tmp.multiply(RHS_i, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag() = Tmp;
      Tmp = LHS_i;
      Tmp.multiply(RHS_r, APFloat::rmNearestTiesToEven);
      Result.getComplexFloatImag().add(Tmp, APFloat::rmNearestTiesToEven);
    } else {
      ComplexValue LHS = Result;
      Result.getComplexIntReal() =
        (LHS.getComplexIntReal() * RHS.getComplexIntReal() -
         LHS.getComplexIntImag() * RHS.getComplexIntImag());
      Result.getComplexIntImag() =
        (LHS.getComplexIntReal() * RHS.getComplexIntImag() +
         LHS.getComplexIntImag() * RHS.getComplexIntReal());
    }
    break;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Top level Expr::Evaluate method.
//===----------------------------------------------------------------------===//

/// Evaluate - Return true if this is a constant which we can fold using
/// any crazy technique (that has nothing to do with language standards) that
/// we want to.  If this function returns true, it returns the folded constant
/// in Result.
bool Expr::Evaluate(EvalResult &Result, ASTContext &Ctx) const {
  const Expr *E = this;
  EvalInfo Info(Ctx, Result);
  if (E->getType()->isVectorType()) {
    if (!EvaluateVector(E, Info.EvalResult.Val, Info))
      return false;
  } else if (E->getType()->isIntegerType()) {
    if (!IntExprEvaluator(Info, Info.EvalResult.Val).Visit(const_cast<Expr*>(E)))
      return false;
    if (Result.Val.isLValue() && !IsGlobalLValue(Result.Val.getLValueBase()))
      return false;
  } else if (E->getType()->hasPointerRepresentation()) {
    LValue LV;
    if (!EvaluatePointer(E, LV, Info))
      return false;
    if (!IsGlobalLValue(LV.Base))
      return false;
    LV.moveInto(Info.EvalResult.Val);
  } else if (E->getType()->isRealFloatingType()) {
    llvm::APFloat F(0.0);
    if (!EvaluateFloat(E, F, Info))
      return false;

    Info.EvalResult.Val = APValue(F);
  } else if (E->getType()->isAnyComplexType()) {
    ComplexValue C;
    if (!EvaluateComplex(E, C, Info))
      return false;
    C.moveInto(Info.EvalResult.Val);
  } else
    return false;

  return true;
}

bool Expr::EvaluateAsBooleanCondition(bool &Result, ASTContext &Ctx) const {
  EvalResult Scratch;
  EvalInfo Info(Ctx, Scratch);

  return HandleConversionToBool(this, Result, Info);
}

bool Expr::EvaluateAsLValue(EvalResult &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  LValue LV;
  if (EvaluateLValue(this, LV, Info) &&
      !Result.HasSideEffects &&
      IsGlobalLValue(LV.Base)) {
    LV.moveInto(Result.Val);
    return true;
  }
  return false;
}

bool Expr::EvaluateAsAnyLValue(EvalResult &Result, ASTContext &Ctx) const {
  EvalInfo Info(Ctx, Result);

  LValue LV;
  if (EvaluateLValue(this, LV, Info)) {
    LV.moveInto(Result.Val);
    return true;
  }
  return false;
}

/// isEvaluatable - Call Evaluate to see if this expression can be constant
/// folded, but discard the result.
bool Expr::isEvaluatable(ASTContext &Ctx) const {
  EvalResult Result;
  return Evaluate(Result, Ctx) && !Result.HasSideEffects;
}

bool Expr::HasSideEffects(ASTContext &Ctx) const {
  Expr::EvalResult Result;
  EvalInfo Info(Ctx, Result);
  return HasSideEffect(Info).Visit(const_cast<Expr*>(this));
}

APSInt Expr::EvaluateAsInt(ASTContext &Ctx) const {
  EvalResult EvalResult;
  bool Result = Evaluate(EvalResult, Ctx);
  Result = Result;
  assert(Result && "Could not evaluate expression");
  assert(EvalResult.Val.isInt() && "Expression did not evaluate to integer");

  return EvalResult.Val.getInt();
}

 bool Expr::EvalResult::isGlobalLValue() const {
   assert(Val.isLValue());
   return IsGlobalLValue(Val.getLValueBase());
 }


/// isIntegerConstantExpr - this recursive routine will test if an expression is
/// an integer constant expression.

/// FIXME: Pass up a reason why! Invalid operation in i-c-e, division by zero,
/// comma, etc
///
/// FIXME: Handle offsetof.  Two things to do:  Handle GCC's __builtin_offsetof
/// to support gcc 4.0+  and handle the idiom GCC recognizes with a null pointer
/// cast+dereference.

// CheckICE - This function does the fundamental ICE checking: the returned
// ICEDiag contains a Val of 0, 1, or 2, and a possibly null SourceLocation.
// Note that to reduce code duplication, this helper does no evaluation
// itself; the caller checks whether the expression is evaluatable, and
// in the rare cases where CheckICE actually cares about the evaluated
// value, it calls into Evalute.
//
// Meanings of Val:
// 0: This expression is an ICE if it can be evaluated by Evaluate.
// 1: This expression is not an ICE, but if it isn't evaluated, it's
//    a legal subexpression for an ICE. This return value is used to handle
//    the comma operator in C99 mode.
// 2: This expression is not an ICE, and is not a legal subexpression for one.

namespace {

struct ICEDiag {
  unsigned Val;
  SourceLocation Loc;

  public:
  ICEDiag(unsigned v, SourceLocation l) : Val(v), Loc(l) {}
  ICEDiag() : Val(0) {}
};

}

static ICEDiag NoDiag() { return ICEDiag(); }

static ICEDiag CheckEvalInICE(const Expr* E, ASTContext &Ctx) {
  Expr::EvalResult EVResult;
  if (!E->Evaluate(EVResult, Ctx) || EVResult.HasSideEffects ||
      !EVResult.Val.isInt()) {
    return ICEDiag(2, E->getLocStart());
  }
  return NoDiag();
}

static ICEDiag CheckICE(const Expr* E, ASTContext &Ctx) {
  assert(!E->isValueDependent() && "Should not see value dependent exprs!");
  if (!E->getType()->isIntegralOrEnumerationType()) {
    return ICEDiag(2, E->getLocStart());
  }

  switch (E->getStmtClass()) {
#define STMT(Node, Base) case Expr::Node##Class:
#define EXPR(Node, Base)
#include "clang/AST/StmtNodes.inc"
  case Expr::PredefinedExprClass:
  case Expr::FloatingLiteralClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::StringLiteralClass:
  case Expr::ArraySubscriptExprClass:
  case Expr::MemberExprClass:
  case Expr::CompoundAssignOperatorClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::InitListExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::ParenListExprClass:
  case Expr::VAArgExprClass:
  case Expr::AddrLabelExprClass:
  case Expr::StmtExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXTypeidExprClass:
  case Expr::CXXNullPtrLiteralExprClass:
  case Expr::CXXThisExprClass:
  case Expr::CXXThrowExprClass:
  case Expr::CXXNewExprClass:
  case Expr::CXXDeleteExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::UnresolvedLookupExprClass:
  case Expr::DependentScopeDeclRefExprClass:
  case Expr::CXXConstructExprClass:
  case Expr::CXXBindTemporaryExprClass:
  case Expr::CXXExprWithTemporariesClass:
  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXUnresolvedConstructExprClass:
  case Expr::CXXDependentScopeMemberExprClass:
  case Expr::UnresolvedMemberExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCImplicitSetterGetterRefExprClass:
  case Expr::ObjCSuperExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::BlockExprClass:
  case Expr::BlockDeclRefExprClass:
  case Expr::NoStmtClass:
    return ICEDiag(2, E->getLocStart());

  case Expr::GNUNullExprClass:
    // GCC considers the GNU __null value to be an integral constant expression.
    return NoDiag();

  case Expr::ParenExprClass:
    return CheckICE(cast<ParenExpr>(E)->getSubExpr(), Ctx);
  case Expr::IntegerLiteralClass:
  case Expr::CharacterLiteralClass:
  case Expr::CXXBoolLiteralExprClass:
  case Expr::CXXScalarValueInitExprClass:
  case Expr::TypesCompatibleExprClass:
  case Expr::UnaryTypeTraitExprClass:
    return NoDiag();
  case Expr::CallExprClass:
  case Expr::CXXOperatorCallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    if (CE->isBuiltinCall(Ctx))
      return CheckEvalInICE(E, Ctx);
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::DeclRefExprClass:
    if (isa<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return NoDiag();
    if (Ctx.getLangOptions().CPlusPlus &&
        E->getType().getCVRQualifiers() == Qualifiers::Const) {
      const NamedDecl *D = cast<DeclRefExpr>(E)->getDecl();

      // Parameter variables are never constants.  Without this check,
      // getAnyInitializer() can find a default argument, which leads
      // to chaos.
      if (isa<ParmVarDecl>(D))
        return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());

      // C++ 7.1.5.1p2
      //   A variable of non-volatile const-qualified integral or enumeration
      //   type initialized by an ICE can be used in ICEs.
      if (const VarDecl *Dcl = dyn_cast<VarDecl>(D)) {
        Qualifiers Quals = Ctx.getCanonicalType(Dcl->getType()).getQualifiers();
        if (Quals.hasVolatile() || !Quals.hasConst())
          return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
        
        // Look for a declaration of this variable that has an initializer.
        const VarDecl *ID = 0;
        const Expr *Init = Dcl->getAnyInitializer(ID);
        if (Init) {
          if (ID->isInitKnownICE()) {
            // We have already checked whether this subexpression is an
            // integral constant expression.
            if (ID->isInitICE())
              return NoDiag();
            else
              return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          // It's an ICE whether or not the definition we found is
          // out-of-line.  See DR 721 and the discussion in Clang PR
          // 6206 for details.

          if (Dcl->isCheckingICE()) {
            return ICEDiag(2, cast<DeclRefExpr>(E)->getLocation());
          }

          Dcl->setCheckingICE();
          ICEDiag Result = CheckICE(Init, Ctx);
          // Cache the result of the ICE test.
          Dcl->setInitKnownICE(Result.Val == 0);
          return Result;
        }
      }
    }
    return ICEDiag(2, E->getLocStart());
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);
    switch (Exp->getOpcode()) {
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
    case UO_AddrOf:
    case UO_Deref:
      return ICEDiag(2, E->getLocStart());
    case UO_Extension:
    case UO_LNot:
    case UO_Plus:
    case UO_Minus:
    case UO_Not:
    case UO_Real:
    case UO_Imag:
      return CheckICE(Exp->getSubExpr(), Ctx);
    }
    
    // OffsetOf falls through here.
  }
  case Expr::OffsetOfExprClass: {
      // Note that per C99, offsetof must be an ICE. And AFAIK, using
      // Evaluate matches the proposed gcc behavior for cases like
      // "offsetof(struct s{int x[4];}, x[!.0])".  This doesn't affect
      // compliance: we should warn earlier for offsetof expressions with
      // array subscripts that aren't ICEs, and if the array subscripts
      // are ICEs, the value of the offsetof must be an integer constant.
      return CheckEvalInICE(E, Ctx);
  }
  case Expr::SizeOfAlignOfExprClass: {
    const SizeOfAlignOfExpr *Exp = cast<SizeOfAlignOfExpr>(E);
    if (Exp->isSizeOf() && Exp->getTypeOfArgument()->isVariableArrayType())
      return ICEDiag(2, E->getLocStart());
    return NoDiag();
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(E);
    switch (Exp->getOpcode()) {
    case BO_PtrMemD:
    case BO_PtrMemI:
    case BO_Assign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      return ICEDiag(2, E->getLocStart());

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
    case BO_And:
    case BO_Xor:
    case BO_Or:
    case BO_Comma: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (Exp->getOpcode() == BO_Div ||
          Exp->getOpcode() == BO_Rem) {
        // Evaluate gives an error for undefined Div/Rem, so make sure
        // we don't evaluate one.
        if (LHSResult.Val != 2 && RHSResult.Val != 2) {
          llvm::APSInt REval = Exp->getRHS()->EvaluateAsInt(Ctx);
          if (REval == 0)
            return ICEDiag(1, E->getLocStart());
          if (REval.isSigned() && REval.isAllOnesValue()) {
            llvm::APSInt LEval = Exp->getLHS()->EvaluateAsInt(Ctx);
            if (LEval.isMinSignedValue())
              return ICEDiag(1, E->getLocStart());
          }
        }
      }
      if (Exp->getOpcode() == BO_Comma) {
        if (Ctx.getLangOptions().C99) {
          // C99 6.6p3 introduces a strange edge case: comma can be in an ICE
          // if it isn't evaluated.
          if (LHSResult.Val == 0 && RHSResult.Val == 0)
            return ICEDiag(1, E->getLocStart());
        } else {
          // In both C89 and C++, commas in ICEs are illegal.
          return ICEDiag(2, E->getLocStart());
        }
      }
      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    case BO_LAnd:
    case BO_LOr: {
      ICEDiag LHSResult = CheckICE(Exp->getLHS(), Ctx);
      ICEDiag RHSResult = CheckICE(Exp->getRHS(), Ctx);
      if (LHSResult.Val == 0 && RHSResult.Val == 1) {
        // Rare case where the RHS has a comma "side-effect"; we need
        // to actually check the condition to see whether the side
        // with the comma is evaluated.
        if ((Exp->getOpcode() == BO_LAnd) !=
            (Exp->getLHS()->EvaluateAsInt(Ctx) == 0))
          return RHSResult;
        return NoDiag();
      }

      if (LHSResult.Val >= RHSResult.Val)
        return LHSResult;
      return RHSResult;
    }
    }
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(E)->getSubExpr();
    if (SubExpr->getType()->isIntegralOrEnumerationType())
      return CheckICE(SubExpr, Ctx);
    if (isa<FloatingLiteral>(SubExpr->IgnoreParens()))
      return NoDiag();
    return ICEDiag(2, E->getLocStart());
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(E);
    // If the condition (ignoring parens) is a __builtin_constant_p call,
    // then only the true side is actually considered in an integer constant
    // expression, and it is fully evaluated.  This is an important GNU
    // extension.  See GCC PR38377 for discussion.
    if (const CallExpr *CallCE
        = dyn_cast<CallExpr>(Exp->getCond()->IgnoreParenCasts()))
      if (CallCE->isBuiltinCall(Ctx) == Builtin::BI__builtin_constant_p) {
        Expr::EvalResult EVResult;
        if (!E->Evaluate(EVResult, Ctx) || EVResult.HasSideEffects ||
            !EVResult.Val.isInt()) {
          return ICEDiag(2, E->getLocStart());
        }
        return NoDiag();
      }
    ICEDiag CondResult = CheckICE(Exp->getCond(), Ctx);
    ICEDiag TrueResult = CheckICE(Exp->getTrueExpr(), Ctx);
    ICEDiag FalseResult = CheckICE(Exp->getFalseExpr(), Ctx);
    if (CondResult.Val == 2)
      return CondResult;
    if (TrueResult.Val == 2)
      return TrueResult;
    if (FalseResult.Val == 2)
      return FalseResult;
    if (CondResult.Val == 1)
      return CondResult;
    if (TrueResult.Val == 0 && FalseResult.Val == 0)
      return NoDiag();
    // Rare case where the diagnostics depend on which side is evaluated
    // Note that if we get here, CondResult is 0, and at least one of
    // TrueResult and FalseResult is non-zero.
    if (Exp->getCond()->EvaluateAsInt(Ctx) == 0) {
      return FalseResult;
    }
    return TrueResult;
  }
  case Expr::CXXDefaultArgExprClass:
    return CheckICE(cast<CXXDefaultArgExpr>(E)->getExpr(), Ctx);
  case Expr::ChooseExprClass: {
    return CheckICE(cast<ChooseExpr>(E)->getChosenSubExpr(Ctx), Ctx);
  }
  }

  // Silence a GCC warning
  return ICEDiag(2, E->getLocStart());
}

bool Expr::isIntegerConstantExpr(llvm::APSInt &Result, ASTContext &Ctx,
                                 SourceLocation *Loc, bool isEvaluated) const {
  ICEDiag d = CheckICE(this, Ctx);
  if (d.Val != 0) {
    if (Loc) *Loc = d.Loc;
    return false;
  }
  EvalResult EvalResult;
  if (!Evaluate(EvalResult, Ctx))
    llvm_unreachable("ICE cannot be evaluated!");
  assert(!EvalResult.HasSideEffects && "ICE with side effects!");
  assert(EvalResult.Val.isInt() && "ICE that isn't integer!");
  Result = EvalResult.Val.getInt();
  return true;
}
