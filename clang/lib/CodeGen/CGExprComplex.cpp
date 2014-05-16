//===--- CGExprComplex.cpp - Emit LLVM Code for Complex Exprs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with complex types as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include <algorithm>
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Complex Expression Emitter
//===----------------------------------------------------------------------===//

typedef CodeGenFunction::ComplexPairTy ComplexPairTy;

/// Return the complex type that we are meant to emit.
static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type)) {
    return comp;
  } else {
    return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
  }
}

namespace  {
class ComplexExprEmitter
  : public StmtVisitor<ComplexExprEmitter, ComplexPairTy> {
  CodeGenFunction &CGF;
  CGBuilderTy &Builder;
  bool IgnoreReal;
  bool IgnoreImag;
public:
  ComplexExprEmitter(CodeGenFunction &cgf, bool ir=false, bool ii=false)
    : CGF(cgf), Builder(CGF.Builder), IgnoreReal(ir), IgnoreImag(ii) {
  }


  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool TestAndClearIgnoreReal() {
    bool I = IgnoreReal;
    IgnoreReal = false;
    return I;
  }
  bool TestAndClearIgnoreImag() {
    bool I = IgnoreImag;
    IgnoreImag = false;
    return I;
  }

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  ComplexPairTy EmitLoadOfLValue(const Expr *E) {
    return EmitLoadOfLValue(CGF.EmitLValue(E), E->getExprLoc());
  }

  ComplexPairTy EmitLoadOfLValue(LValue LV, SourceLocation Loc);

  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void EmitStoreOfComplex(ComplexPairTy Val, LValue LV, bool isInit);

  /// EmitComplexToComplexCast - Emit a cast from complex value Val to DestType.
  ComplexPairTy EmitComplexToComplexCast(ComplexPairTy Val, QualType SrcType,
                                         QualType DestType);
  /// EmitComplexToComplexCast - Emit a cast from scalar value Val to DestType.
  ComplexPairTy EmitScalarToComplexCast(llvm::Value *Val, QualType SrcType,
                                        QualType DestType);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  ComplexPairTy Visit(Expr *E) {
    return StmtVisitor<ComplexExprEmitter, ComplexPairTy>::Visit(E);
  }
    
  ComplexPairTy VisitStmt(Stmt *S) {
    S->dump(CGF.getContext().getSourceManager());
    llvm_unreachable("Stmt can't have complex result type!");
  }
  ComplexPairTy VisitExpr(Expr *S);
  ComplexPairTy VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr());}
  ComplexPairTy VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    return Visit(GE->getResultExpr());
  }
  ComplexPairTy VisitImaginaryLiteral(const ImaginaryLiteral *IL);
  ComplexPairTy
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *PE) {
    return Visit(PE->getReplacement());
  }

  // l-values.
  ComplexPairTy VisitDeclRefExpr(DeclRefExpr *E) {
    if (CodeGenFunction::ConstantEmission result = CGF.tryEmitAsConstant(E)) {
      if (result.isReference())
        return EmitLoadOfLValue(result.getReferenceLValue(CGF, E),
                                E->getExprLoc());

      llvm::Constant *pair = result.getValue();
      return ComplexPairTy(pair->getAggregateElement(0U),
                           pair->getAggregateElement(1U));
    }
    return EmitLoadOfLValue(E);
  }
  ComplexPairTy VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    return EmitLoadOfLValue(E);
  }
  ComplexPairTy VisitObjCMessageExpr(ObjCMessageExpr *E) {
    return CGF.EmitObjCMessageExpr(E).getComplexVal();
  }
  ComplexPairTy VisitArraySubscriptExpr(Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitMemberExpr(const Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    if (E->isGLValue())
      return EmitLoadOfLValue(CGF.getOpaqueLValueMapping(E), E->getExprLoc());
    return CGF.getOpaqueRValueMapping(E).getComplexVal();
  }

  ComplexPairTy VisitPseudoObjectExpr(PseudoObjectExpr *E) {
    return CGF.EmitPseudoObjectRValue(E).getComplexVal();
  }

  // FIXME: CompoundLiteralExpr

  ComplexPairTy EmitCast(CastExpr::CastKind CK, Expr *Op, QualType DestTy);
  ComplexPairTy VisitImplicitCastExpr(ImplicitCastExpr *E) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    return EmitCast(E->getCastKind(), E->getSubExpr(), E->getType());
  }
  ComplexPairTy VisitCastExpr(CastExpr *E) {
    return EmitCast(E->getCastKind(), E->getSubExpr(), E->getType());
  }
  ComplexPairTy VisitCallExpr(const CallExpr *E);
  ComplexPairTy VisitStmtExpr(const StmtExpr *E);

  // Operators.
  ComplexPairTy VisitPrePostIncDec(const UnaryOperator *E,
                                   bool isInc, bool isPre) {
    LValue LV = CGF.EmitLValue(E->getSubExpr());
    return CGF.EmitComplexPrePostIncDec(E, LV, isInc, isPre);
  }
  ComplexPairTy VisitUnaryPostDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, false);
  }
  ComplexPairTy VisitUnaryPostInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, false);
  }
  ComplexPairTy VisitUnaryPreDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, true);
  }
  ComplexPairTy VisitUnaryPreInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, true);
  }
  ComplexPairTy VisitUnaryDeref(const Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitUnaryPlus     (const UnaryOperator *E) {
    TestAndClearIgnoreReal();
    TestAndClearIgnoreImag();
    return Visit(E->getSubExpr());
  }
  ComplexPairTy VisitUnaryMinus    (const UnaryOperator *E);
  ComplexPairTy VisitUnaryNot      (const UnaryOperator *E);
  // LNot,Real,Imag never return complex.
  ComplexPairTy VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  ComplexPairTy VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }
  ComplexPairTy VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    CodeGenFunction::CXXDefaultInitExprScope Scope(CGF);
    return Visit(DIE->getExpr());
  }
  ComplexPairTy VisitExprWithCleanups(ExprWithCleanups *E) {
    CGF.enterFullExpression(E);
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    return Visit(E->getSubExpr());
  }
  ComplexPairTy VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
    assert(E->getType()->isAnyComplexType() && "Expected complex type!");
    QualType Elem = E->getType()->castAs<ComplexType>()->getElementType();
    llvm::Constant *Null = llvm::Constant::getNullValue(CGF.ConvertType(Elem));
    return ComplexPairTy(Null, Null);
  }
  ComplexPairTy VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
    assert(E->getType()->isAnyComplexType() && "Expected complex type!");
    QualType Elem = E->getType()->castAs<ComplexType>()->getElementType();
    llvm::Constant *Null =
                       llvm::Constant::getNullValue(CGF.ConvertType(Elem));
    return ComplexPairTy(Null, Null);
  }

  struct BinOpInfo {
    ComplexPairTy LHS;
    ComplexPairTy RHS;
    QualType Ty;  // Computation Type.
  };

  BinOpInfo EmitBinOps(const BinaryOperator *E);
  LValue EmitCompoundAssignLValue(const CompoundAssignOperator *E,
                                  ComplexPairTy (ComplexExprEmitter::*Func)
                                  (const BinOpInfo &),
                                  RValue &Val);
  ComplexPairTy EmitCompoundAssign(const CompoundAssignOperator *E,
                                   ComplexPairTy (ComplexExprEmitter::*Func)
                                   (const BinOpInfo &));

  ComplexPairTy EmitBinAdd(const BinOpInfo &Op);
  ComplexPairTy EmitBinSub(const BinOpInfo &Op);
  ComplexPairTy EmitBinMul(const BinOpInfo &Op);
  ComplexPairTy EmitBinDiv(const BinOpInfo &Op);

  ComplexPairTy VisitBinAdd(const BinaryOperator *E) {
    return EmitBinAdd(EmitBinOps(E));
  }
  ComplexPairTy VisitBinSub(const BinaryOperator *E) {
    return EmitBinSub(EmitBinOps(E));
  }
  ComplexPairTy VisitBinMul(const BinaryOperator *E) {
    return EmitBinMul(EmitBinOps(E));
  }
  ComplexPairTy VisitBinDiv(const BinaryOperator *E) {
    return EmitBinDiv(EmitBinOps(E));
  }

  // Compound assignments.
  ComplexPairTy VisitBinAddAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinAdd);
  }
  ComplexPairTy VisitBinSubAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinSub);
  }
  ComplexPairTy VisitBinMulAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinMul);
  }
  ComplexPairTy VisitBinDivAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue EmitBinAssignLValue(const BinaryOperator *E,
                             ComplexPairTy &Val);
  ComplexPairTy VisitBinAssign     (const BinaryOperator *E);
  ComplexPairTy VisitBinComma      (const BinaryOperator *E);


  ComplexPairTy
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *CO);
  ComplexPairTy VisitChooseExpr(ChooseExpr *CE);

  ComplexPairTy VisitInitListExpr(InitListExpr *E);

  ComplexPairTy VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return EmitLoadOfLValue(E);
  }

  ComplexPairTy VisitVAArgExpr(VAArgExpr *E);

  ComplexPairTy VisitAtomicExpr(AtomicExpr *E) {
    return CGF.EmitAtomicExpr(E).getComplexVal();
  }
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfLValue - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
ComplexPairTy ComplexExprEmitter::EmitLoadOfLValue(LValue lvalue,
                                                   SourceLocation loc) {
  assert(lvalue.isSimple() && "non-simple complex l-value?");
  if (lvalue.getType()->isAtomicType())
    return CGF.EmitAtomicLoad(lvalue, loc).getComplexVal();

  llvm::Value *SrcPtr = lvalue.getAddress();
  bool isVolatile = lvalue.isVolatileQualified();
  unsigned AlignR = lvalue.getAlignment().getQuantity();
  ASTContext &C = CGF.getContext();
  QualType ComplexTy = lvalue.getType();
  unsigned ComplexAlign = C.getTypeAlignInChars(ComplexTy).getQuantity();
  unsigned AlignI = std::min(AlignR, ComplexAlign);

  llvm::Value *Real=0, *Imag=0;

  if (!IgnoreReal || isVolatile) {
    llvm::Value *RealP = Builder.CreateStructGEP(SrcPtr, 0,
                                                 SrcPtr->getName() + ".realp");
    Real = Builder.CreateAlignedLoad(RealP, AlignR, isVolatile,
                                     SrcPtr->getName() + ".real");
  }

  if (!IgnoreImag || isVolatile) {
    llvm::Value *ImagP = Builder.CreateStructGEP(SrcPtr, 1,
                                                 SrcPtr->getName() + ".imagp");
    Imag = Builder.CreateAlignedLoad(ImagP, AlignI, isVolatile,
                                     SrcPtr->getName() + ".imag");
  }
  return ComplexPairTy(Real, Imag);
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::EmitStoreOfComplex(ComplexPairTy Val,
                                            LValue lvalue,
                                            bool isInit) {
  if (lvalue.getType()->isAtomicType())
    return CGF.EmitAtomicStore(RValue::getComplex(Val), lvalue, isInit);

  llvm::Value *Ptr = lvalue.getAddress();
  llvm::Value *RealPtr = Builder.CreateStructGEP(Ptr, 0, "real");
  llvm::Value *ImagPtr = Builder.CreateStructGEP(Ptr, 1, "imag");
  unsigned AlignR = lvalue.getAlignment().getQuantity();
  ASTContext &C = CGF.getContext();
  QualType ComplexTy = lvalue.getType();
  unsigned ComplexAlign = C.getTypeAlignInChars(ComplexTy).getQuantity();
  unsigned AlignI = std::min(AlignR, ComplexAlign);

  Builder.CreateAlignedStore(Val.first, RealPtr, AlignR,
                             lvalue.isVolatileQualified());
  Builder.CreateAlignedStore(Val.second, ImagPtr, AlignI,
                             lvalue.isVolatileQualified());
}



//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

ComplexPairTy ComplexExprEmitter::VisitExpr(Expr *E) {
  CGF.ErrorUnsupported(E, "complex expression");
  llvm::Type *EltTy =
    CGF.ConvertType(getComplexType(E->getType())->getElementType());
  llvm::Value *U = llvm::UndefValue::get(EltTy);
  return ComplexPairTy(U, U);
}

ComplexPairTy ComplexExprEmitter::
VisitImaginaryLiteral(const ImaginaryLiteral *IL) {
  llvm::Value *Imag = CGF.EmitScalarExpr(IL->getSubExpr());
  return ComplexPairTy(llvm::Constant::getNullValue(Imag->getType()), Imag);
}


ComplexPairTy ComplexExprEmitter::VisitCallExpr(const CallExpr *E) {
  if (E->getCallReturnType()->isReferenceType())
    return EmitLoadOfLValue(E);

  return CGF.EmitCallExpr(E).getComplexVal();
}

ComplexPairTy ComplexExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  CodeGenFunction::StmtExprEvaluation eval(CGF);
  llvm::Value *RetAlloca = CGF.EmitCompoundStmt(*E->getSubStmt(), true);
  assert(RetAlloca && "Expected complex return value");
  return EmitLoadOfLValue(CGF.MakeAddrLValue(RetAlloca, E->getType()),
                          E->getExprLoc());
}

/// EmitComplexToComplexCast - Emit a cast from complex value Val to DestType.
ComplexPairTy ComplexExprEmitter::EmitComplexToComplexCast(ComplexPairTy Val,
                                                           QualType SrcType,
                                                           QualType DestType) {
  // Get the src/dest element type.
  SrcType = SrcType->castAs<ComplexType>()->getElementType();
  DestType = DestType->castAs<ComplexType>()->getElementType();

  // C99 6.3.1.6: When a value of complex type is converted to another
  // complex type, both the real and imaginary parts follow the conversion
  // rules for the corresponding real types.
  Val.first = CGF.EmitScalarConversion(Val.first, SrcType, DestType);
  Val.second = CGF.EmitScalarConversion(Val.second, SrcType, DestType);
  return Val;
}

ComplexPairTy ComplexExprEmitter::EmitScalarToComplexCast(llvm::Value *Val,
                                                          QualType SrcType,
                                                          QualType DestType) {
  // Convert the input element to the element type of the complex.
  DestType = DestType->castAs<ComplexType>()->getElementType();
  Val = CGF.EmitScalarConversion(Val, SrcType, DestType);

  // Return (realval, 0).
  return ComplexPairTy(Val, llvm::Constant::getNullValue(Val->getType()));
}

ComplexPairTy ComplexExprEmitter::EmitCast(CastExpr::CastKind CK, Expr *Op, 
                                           QualType DestTy) {
  switch (CK) {
  case CK_Dependent: llvm_unreachable("dependent cast kind in IR gen!");

  // Atomic to non-atomic casts may be more than a no-op for some platforms and
  // for some types.
  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_LValueToRValue:
  case CK_UserDefinedConversion:
    return Visit(Op);

  case CK_LValueBitCast: {
    LValue origLV = CGF.EmitLValue(Op);
    llvm::Value *V = origLV.getAddress();
    V = Builder.CreateBitCast(V, 
                    CGF.ConvertType(CGF.getContext().getPointerType(DestTy)));
    return EmitLoadOfLValue(CGF.MakeAddrLValue(V, DestTy,
                                               origLV.getAlignment()),
                            Op->getExprLoc());
  }

  case CK_BitCast:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_BuiltinFnToFnPtr:
  case CK_ZeroToOCLEvent:
  case CK_AddressSpaceConversion:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_FloatingRealToComplex:
  case CK_IntegralRealToComplex:
    return EmitScalarToComplexCast(CGF.EmitScalarExpr(Op),
                                   Op->getType(), DestTy);

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
    return EmitComplexToComplexCast(Visit(Op), Op->getType(), DestTy);
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

ComplexPairTy ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  ComplexPairTy Op = Visit(E->getSubExpr());

  llvm::Value *ResR, *ResI;
  if (Op.first->getType()->isFloatingPointTy()) {
    ResR = Builder.CreateFNeg(Op.first,  "neg.r");
    ResI = Builder.CreateFNeg(Op.second, "neg.i");
  } else {
    ResR = Builder.CreateNeg(Op.first,  "neg.r");
    ResI = Builder.CreateNeg(Op.second, "neg.i");
  }
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  // ~(a+ib) = a + i*-b
  ComplexPairTy Op = Visit(E->getSubExpr());
  llvm::Value *ResI;
  if (Op.second->getType()->isFloatingPointTy())
    ResI = Builder.CreateFNeg(Op.second, "conj.i");
  else
    ResI = Builder.CreateNeg(Op.second, "conj.i");

  return ComplexPairTy(Op.first, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinAdd(const BinOpInfo &Op) {
  llvm::Value *ResR, *ResI;

  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    ResR = Builder.CreateFAdd(Op.LHS.first,  Op.RHS.first,  "add.r");
    ResI = Builder.CreateFAdd(Op.LHS.second, Op.RHS.second, "add.i");
  } else {
    ResR = Builder.CreateAdd(Op.LHS.first,  Op.RHS.first,  "add.r");
    ResI = Builder.CreateAdd(Op.LHS.second, Op.RHS.second, "add.i");
  }
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinSub(const BinOpInfo &Op) {
  llvm::Value *ResR, *ResI;
  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    ResR = Builder.CreateFSub(Op.LHS.first,  Op.RHS.first,  "sub.r");
    ResI = Builder.CreateFSub(Op.LHS.second, Op.RHS.second, "sub.i");
  } else {
    ResR = Builder.CreateSub(Op.LHS.first,  Op.RHS.first,  "sub.r");
    ResI = Builder.CreateSub(Op.LHS.second, Op.RHS.second, "sub.i");
  }
  return ComplexPairTy(ResR, ResI);
}


ComplexPairTy ComplexExprEmitter::EmitBinMul(const BinOpInfo &Op) {
  using llvm::Value;
  Value *ResR, *ResI;

  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    Value *ResRl = Builder.CreateFMul(Op.LHS.first, Op.RHS.first, "mul.rl");
    Value *ResRr = Builder.CreateFMul(Op.LHS.second, Op.RHS.second,"mul.rr");
    ResR  = Builder.CreateFSub(ResRl, ResRr, "mul.r");

    Value *ResIl = Builder.CreateFMul(Op.LHS.second, Op.RHS.first, "mul.il");
    Value *ResIr = Builder.CreateFMul(Op.LHS.first, Op.RHS.second, "mul.ir");
    ResI  = Builder.CreateFAdd(ResIl, ResIr, "mul.i");
  } else {
    Value *ResRl = Builder.CreateMul(Op.LHS.first, Op.RHS.first, "mul.rl");
    Value *ResRr = Builder.CreateMul(Op.LHS.second, Op.RHS.second,"mul.rr");
    ResR  = Builder.CreateSub(ResRl, ResRr, "mul.r");

    Value *ResIl = Builder.CreateMul(Op.LHS.second, Op.RHS.first, "mul.il");
    Value *ResIr = Builder.CreateMul(Op.LHS.first, Op.RHS.second, "mul.ir");
    ResI  = Builder.CreateAdd(ResIl, ResIr, "mul.i");
  }
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinDiv(const BinOpInfo &Op) {
  llvm::Value *LHSr = Op.LHS.first, *LHSi = Op.LHS.second;
  llvm::Value *RHSr = Op.RHS.first, *RHSi = Op.RHS.second;


  llvm::Value *DSTr, *DSTi;
  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    // (a+ib) / (c+id) = ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
    llvm::Value *Tmp1 = Builder.CreateFMul(LHSr, RHSr); // a*c
    llvm::Value *Tmp2 = Builder.CreateFMul(LHSi, RHSi); // b*d
    llvm::Value *Tmp3 = Builder.CreateFAdd(Tmp1, Tmp2); // ac+bd

    llvm::Value *Tmp4 = Builder.CreateFMul(RHSr, RHSr); // c*c
    llvm::Value *Tmp5 = Builder.CreateFMul(RHSi, RHSi); // d*d
    llvm::Value *Tmp6 = Builder.CreateFAdd(Tmp4, Tmp5); // cc+dd

    llvm::Value *Tmp7 = Builder.CreateFMul(LHSi, RHSr); // b*c
    llvm::Value *Tmp8 = Builder.CreateFMul(LHSr, RHSi); // a*d
    llvm::Value *Tmp9 = Builder.CreateFSub(Tmp7, Tmp8); // bc-ad

    DSTr = Builder.CreateFDiv(Tmp3, Tmp6);
    DSTi = Builder.CreateFDiv(Tmp9, Tmp6);
  } else {
    // (a+ib) / (c+id) = ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
    llvm::Value *Tmp1 = Builder.CreateMul(LHSr, RHSr); // a*c
    llvm::Value *Tmp2 = Builder.CreateMul(LHSi, RHSi); // b*d
    llvm::Value *Tmp3 = Builder.CreateAdd(Tmp1, Tmp2); // ac+bd

    llvm::Value *Tmp4 = Builder.CreateMul(RHSr, RHSr); // c*c
    llvm::Value *Tmp5 = Builder.CreateMul(RHSi, RHSi); // d*d
    llvm::Value *Tmp6 = Builder.CreateAdd(Tmp4, Tmp5); // cc+dd

    llvm::Value *Tmp7 = Builder.CreateMul(LHSi, RHSr); // b*c
    llvm::Value *Tmp8 = Builder.CreateMul(LHSr, RHSi); // a*d
    llvm::Value *Tmp9 = Builder.CreateSub(Tmp7, Tmp8); // bc-ad

    if (Op.Ty->castAs<ComplexType>()->getElementType()->isUnsignedIntegerType()) {
      DSTr = Builder.CreateUDiv(Tmp3, Tmp6);
      DSTi = Builder.CreateUDiv(Tmp9, Tmp6);
    } else {
      DSTr = Builder.CreateSDiv(Tmp3, Tmp6);
      DSTi = Builder.CreateSDiv(Tmp9, Tmp6);
    }
  }

  return ComplexPairTy(DSTr, DSTi);
}

ComplexExprEmitter::BinOpInfo
ComplexExprEmitter::EmitBinOps(const BinaryOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  BinOpInfo Ops;
  Ops.LHS = Visit(E->getLHS());
  Ops.RHS = Visit(E->getRHS());
  Ops.Ty = E->getType();
  return Ops;
}


LValue ComplexExprEmitter::
EmitCompoundAssignLValue(const CompoundAssignOperator *E,
          ComplexPairTy (ComplexExprEmitter::*Func)(const BinOpInfo&),
                         RValue &Val) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  QualType LHSTy = E->getLHS()->getType();

  BinOpInfo OpInfo;

  // Load the RHS and LHS operands.
  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen a little.
  OpInfo.Ty = E->getComputationResultType();

  // The RHS should have been converted to the computation type.
  assert(OpInfo.Ty->isAnyComplexType());
  assert(CGF.getContext().hasSameUnqualifiedType(OpInfo.Ty,
                                                 E->getRHS()->getType()));
  OpInfo.RHS = Visit(E->getRHS());
  
  LValue LHS = CGF.EmitLValue(E->getLHS());

  // Load from the l-value and convert it.
  if (LHSTy->isAnyComplexType()) {
    ComplexPairTy LHSVal = EmitLoadOfLValue(LHS, E->getExprLoc());
    OpInfo.LHS = EmitComplexToComplexCast(LHSVal, LHSTy, OpInfo.Ty);
  } else {
    llvm::Value *LHSVal = CGF.EmitLoadOfScalar(LHS, E->getExprLoc());
    OpInfo.LHS = EmitScalarToComplexCast(LHSVal, LHSTy, OpInfo.Ty);
  }

  // Expand the binary operator.
  ComplexPairTy Result = (this->*Func)(OpInfo);

  // Truncate the result and store it into the LHS lvalue.
  if (LHSTy->isAnyComplexType()) {
    ComplexPairTy ResVal = EmitComplexToComplexCast(Result, OpInfo.Ty, LHSTy);
    EmitStoreOfComplex(ResVal, LHS, /*isInit*/ false);
    Val = RValue::getComplex(ResVal);
  } else {
    llvm::Value *ResVal =
        CGF.EmitComplexToScalarConversion(Result, OpInfo.Ty, LHSTy);
    CGF.EmitStoreOfScalar(ResVal, LHS, /*isInit*/ false);
    Val = RValue::get(ResVal);
  }

  return LHS;
}

// Compound assignments.
ComplexPairTy ComplexExprEmitter::
EmitCompoundAssign(const CompoundAssignOperator *E,
                   ComplexPairTy (ComplexExprEmitter::*Func)(const BinOpInfo&)){
  RValue Val;
  LValue LV = EmitCompoundAssignLValue(E, Func, Val);

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return Val.getComplexVal();

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LV.isVolatileQualified())
    return Val.getComplexVal();

  return EmitLoadOfLValue(LV, E->getExprLoc());
}

LValue ComplexExprEmitter::EmitBinAssignLValue(const BinaryOperator *E,
                                               ComplexPairTy &Val) {
  assert(CGF.getContext().hasSameUnqualifiedType(E->getLHS()->getType(), 
                                                 E->getRHS()->getType()) &&
         "Invalid assignment");
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();

  // Emit the RHS.  __block variables need the RHS evaluated first.
  Val = Visit(E->getRHS());

  // Compute the address to store into.
  LValue LHS = CGF.EmitLValue(E->getLHS());

  // Store the result value into the LHS lvalue.
  EmitStoreOfComplex(Val, LHS, /*isInit*/ false);

  return LHS;
}

ComplexPairTy ComplexExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  ComplexPairTy Val;
  LValue LV = EmitBinAssignLValue(E, Val);

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return Val;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LV.isVolatileQualified())
    return Val;

  return EmitLoadOfLValue(LV, E->getExprLoc());
}

ComplexPairTy ComplexExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitIgnoredExpr(E->getLHS());
  return Visit(E->getRHS());
}

ComplexPairTy ComplexExprEmitter::
VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  llvm::BasicBlock *LHSBlock = CGF.createBasicBlock("cond.true");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("cond.false");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("cond.end");

  // Bind the common expression if necessary.
  CodeGenFunction::OpaqueValueMapping binding(CGF, E);

  CodeGenFunction::ConditionalEvaluation eval(CGF);
  CGF.EmitBranchOnBoolExpr(E->getCond(), LHSBlock, RHSBlock);

  eval.begin(CGF);
  CGF.EmitBlock(LHSBlock);
  ComplexPairTy LHS = Visit(E->getTrueExpr());
  LHSBlock = Builder.GetInsertBlock();
  CGF.EmitBranch(ContBlock);
  eval.end(CGF);

  eval.begin(CGF);
  CGF.EmitBlock(RHSBlock);
  ComplexPairTy RHS = Visit(E->getFalseExpr());
  RHSBlock = Builder.GetInsertBlock();
  CGF.EmitBlock(ContBlock);
  eval.end(CGF);

  // Create a PHI node for the real part.
  llvm::PHINode *RealPN = Builder.CreatePHI(LHS.first->getType(), 2, "cond.r");
  RealPN->addIncoming(LHS.first, LHSBlock);
  RealPN->addIncoming(RHS.first, RHSBlock);

  // Create a PHI node for the imaginary part.
  llvm::PHINode *ImagPN = Builder.CreatePHI(LHS.first->getType(), 2, "cond.i");
  ImagPN->addIncoming(LHS.second, LHSBlock);
  ImagPN->addIncoming(RHS.second, RHSBlock);

  return ComplexPairTy(RealPN, ImagPN);
}

ComplexPairTy ComplexExprEmitter::VisitChooseExpr(ChooseExpr *E) {
  return Visit(E->getChosenSubExpr());
}

ComplexPairTy ComplexExprEmitter::VisitInitListExpr(InitListExpr *E) {
    bool Ignore = TestAndClearIgnoreReal();
    (void)Ignore;
    assert (Ignore == false && "init list ignored");
    Ignore = TestAndClearIgnoreImag();
    (void)Ignore;
    assert (Ignore == false && "init list ignored");

  if (E->getNumInits() == 2) {
    llvm::Value *Real = CGF.EmitScalarExpr(E->getInit(0));
    llvm::Value *Imag = CGF.EmitScalarExpr(E->getInit(1));
    return ComplexPairTy(Real, Imag);
  } else if (E->getNumInits() == 1) {
    return Visit(E->getInit(0));
  }

  // Empty init list intializes to null
  assert(E->getNumInits() == 0 && "Unexpected number of inits");
  QualType Ty = E->getType()->castAs<ComplexType>()->getElementType();
  llvm::Type* LTy = CGF.ConvertType(Ty);
  llvm::Value* zeroConstant = llvm::Constant::getNullValue(LTy);
  return ComplexPairTy(zeroConstant, zeroConstant);
}

ComplexPairTy ComplexExprEmitter::VisitVAArgExpr(VAArgExpr *E) {
  llvm::Value *ArgValue = CGF.EmitVAListRef(E->getSubExpr());
  llvm::Value *ArgPtr = CGF.EmitVAArg(ArgValue, E->getType());

  if (!ArgPtr) {
    CGF.ErrorUnsupported(E, "complex va_arg expression");
    llvm::Type *EltTy =
      CGF.ConvertType(E->getType()->castAs<ComplexType>()->getElementType());
    llvm::Value *U = llvm::UndefValue::get(EltTy);
    return ComplexPairTy(U, U);
  }

  return EmitLoadOfLValue(CGF.MakeNaturalAlignAddrLValue(ArgPtr, E->getType()),
                          E->getExprLoc());
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
ComplexPairTy CodeGenFunction::EmitComplexExpr(const Expr *E, bool IgnoreReal,
                                               bool IgnoreImag) {
  assert(E && getComplexType(E->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this, IgnoreReal, IgnoreImag)
    .Visit(const_cast<Expr*>(E));
}

void CodeGenFunction::EmitComplexExprIntoLValue(const Expr *E, LValue dest,
                                                bool isInit) {
  assert(E && getComplexType(E->getType()) &&
         "Invalid complex expression to emit");
  ComplexExprEmitter Emitter(*this);
  ComplexPairTy Val = Emitter.Visit(const_cast<Expr*>(E));
  Emitter.EmitStoreOfComplex(Val, dest, isInit);
}

/// EmitStoreOfComplex - Store a complex number into the specified l-value.
void CodeGenFunction::EmitStoreOfComplex(ComplexPairTy V, LValue dest,
                                         bool isInit) {
  ComplexExprEmitter(*this).EmitStoreOfComplex(V, dest, isInit);
}

/// EmitLoadOfComplex - Load a complex number from the specified address.
ComplexPairTy CodeGenFunction::EmitLoadOfComplex(LValue src,
                                                 SourceLocation loc) {
  return ComplexExprEmitter(*this).EmitLoadOfLValue(src, loc);
}

LValue CodeGenFunction::EmitComplexAssignmentLValue(const BinaryOperator *E) {
  assert(E->getOpcode() == BO_Assign);
  ComplexPairTy Val; // ignored
  return ComplexExprEmitter(*this).EmitBinAssignLValue(E, Val);
}

typedef ComplexPairTy (ComplexExprEmitter::*CompoundFunc)(
    const ComplexExprEmitter::BinOpInfo &);

static CompoundFunc getComplexOp(BinaryOperatorKind Op) {
  switch (Op) {
  case BO_MulAssign: return &ComplexExprEmitter::EmitBinMul;
  case BO_DivAssign: return &ComplexExprEmitter::EmitBinDiv;
  case BO_SubAssign: return &ComplexExprEmitter::EmitBinSub;
  case BO_AddAssign: return &ComplexExprEmitter::EmitBinAdd;
  default:
    llvm_unreachable("unexpected complex compound assignment");
  }
}

LValue CodeGenFunction::
EmitComplexCompoundAssignmentLValue(const CompoundAssignOperator *E) {
  CompoundFunc Op = getComplexOp(E->getOpcode());
  RValue Val;
  return ComplexExprEmitter(*this).EmitCompoundAssignLValue(E, Op, Val);
}

LValue CodeGenFunction::
EmitScalarCompooundAssignWithComplex(const CompoundAssignOperator *E,
                                     llvm::Value *&Result) {
  CompoundFunc Op = getComplexOp(E->getOpcode());
  RValue Val;
  LValue Ret = ComplexExprEmitter(*this).EmitCompoundAssignLValue(E, Op, Val);
  Result = Val.getScalarVal();
  return Ret;
}
