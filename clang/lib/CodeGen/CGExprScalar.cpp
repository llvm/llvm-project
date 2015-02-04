//===--- CGExprScalar.cpp - Emit LLVM Code for Scalar Exprs ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with scalar LLVM types as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCXXABI.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include <cstdarg>

using namespace clang;
using namespace CodeGen;
using llvm::Value;

//===----------------------------------------------------------------------===//
//                         Scalar Expression Emitter
//===----------------------------------------------------------------------===//

namespace {
struct BinOpInfo {
  Value *LHS;
  Value *RHS;
  QualType Ty;  // Computation Type.
  BinaryOperator::Opcode Opcode; // Opcode of BinOp to perform
  bool FPContractable;
  const Expr *E;      // Entire expr, for error unsupported.  May not be binop.
};

static bool MustVisitNullValue(const Expr *E) {
  // If a null pointer expression's type is the C++0x nullptr_t, then
  // it's not necessarily a simple constant and it must be evaluated
  // for its potential side effects.
  return E->getType()->isNullPtrType();
}

class ScalarExprEmitter
  : public StmtVisitor<ScalarExprEmitter, Value*> {
  CodeGenFunction &CGF;
  CGBuilderTy &Builder;
  bool IgnoreResultAssign;
  llvm::LLVMContext &VMContext;
public:

  ScalarExprEmitter(CodeGenFunction &cgf, bool ira=false)
    : CGF(cgf), Builder(CGF.Builder), IgnoreResultAssign(ira),
      VMContext(cgf.getLLVMContext()) {
  }

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool TestAndClearIgnoreResultAssign() {
    bool I = IgnoreResultAssign;
    IgnoreResultAssign = false;
    return I;
  }

  llvm::Type *ConvertType(QualType T) { return CGF.ConvertType(T); }
  LValue EmitLValue(const Expr *E) { return CGF.EmitLValue(E); }
  LValue EmitCheckedLValue(const Expr *E, CodeGenFunction::TypeCheckKind TCK) {
    return CGF.EmitCheckedLValue(E, TCK);
  }

  void EmitBinOpCheck(ArrayRef<std::pair<Value *, SanitizerKind>> Checks,
                      const BinOpInfo &Info);

  Value *EmitLoadOfLValue(LValue LV, SourceLocation Loc) {
    return CGF.EmitLoadOfLValue(LV, Loc).getScalarVal();
  }

  void EmitLValueAlignmentAssumption(const Expr *E, Value *V) {
    const AlignValueAttr *AVAttr = nullptr;
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      const ValueDecl *VD = DRE->getDecl();

      if (VD->getType()->isReferenceType()) {
        if (const auto *TTy =
            dyn_cast<TypedefType>(VD->getType().getNonReferenceType()))
          AVAttr = TTy->getDecl()->getAttr<AlignValueAttr>();
      } else {
        // Assumptions for function parameters are emitted at the start of the
        // function, so there is no need to repeat that here.
        if (isa<ParmVarDecl>(VD))
          return;

        AVAttr = VD->getAttr<AlignValueAttr>();
      }
    }

    if (!AVAttr)
      if (const auto *TTy =
          dyn_cast<TypedefType>(E->getType()))
        AVAttr = TTy->getDecl()->getAttr<AlignValueAttr>();

    if (!AVAttr)
      return;

    Value *AlignmentValue = CGF.EmitScalarExpr(AVAttr->getAlignment());
    llvm::ConstantInt *AlignmentCI = cast<llvm::ConstantInt>(AlignmentValue);
    CGF.EmitAlignmentAssumption(V, AlignmentCI->getZExtValue());
  }

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  Value *EmitLoadOfLValue(const Expr *E) {
    Value *V = EmitLoadOfLValue(EmitCheckedLValue(E, CodeGenFunction::TCK_Load),
                                E->getExprLoc());

    EmitLValueAlignmentAssumption(E, V);
    return V;
  }

  /// EmitConversionToBool - Convert the specified expression value to a
  /// boolean (i1) truth value.  This is equivalent to "Val != 0".
  Value *EmitConversionToBool(Value *Src, QualType DstTy);

  /// \brief Emit a check that a conversion to or from a floating-point type
  /// does not overflow.
  void EmitFloatConversionCheck(Value *OrigSrc, QualType OrigSrcType,
                                Value *Src, QualType SrcType,
                                QualType DstType, llvm::Type *DstTy);

  /// EmitScalarConversion - Emit a conversion from the specified type to the
  /// specified destination type, both of which are LLVM scalar types.
  Value *EmitScalarConversion(Value *Src, QualType SrcTy, QualType DstTy);

  /// EmitComplexToScalarConversion - Emit a conversion from the specified
  /// complex type to the specified destination type, where the destination type
  /// is an LLVM scalar type.
  Value *EmitComplexToScalarConversion(CodeGenFunction::ComplexPairTy Src,
                                       QualType SrcTy, QualType DstTy);

  /// EmitNullValue - Emit a value that corresponds to null for the given type.
  Value *EmitNullValue(QualType Ty);

  /// EmitFloatToBoolConversion - Perform an FP to boolean conversion.
  Value *EmitFloatToBoolConversion(Value *V) {
    // Compare against 0.0 for fp scalars.
    llvm::Value *Zero = llvm::Constant::getNullValue(V->getType());
    return Builder.CreateFCmpUNE(V, Zero, "tobool");
  }

  /// EmitPointerToBoolConversion - Perform a pointer to boolean conversion.
  Value *EmitPointerToBoolConversion(Value *V) {
    Value *Zero = llvm::ConstantPointerNull::get(
                                      cast<llvm::PointerType>(V->getType()));
    return Builder.CreateICmpNE(V, Zero, "tobool");
  }

  Value *EmitIntToBoolConversion(Value *V) {
    // Because of the type rules of C, we often end up computing a
    // logical value, then zero extending it to int, then wanting it
    // as a logical value again.  Optimize this common case.
    if (llvm::ZExtInst *ZI = dyn_cast<llvm::ZExtInst>(V)) {
      if (ZI->getOperand(0)->getType() == Builder.getInt1Ty()) {
        Value *Result = ZI->getOperand(0);
        // If there aren't any more uses, zap the instruction to save space.
        // Note that there can be more uses, for example if this
        // is the result of an assignment.
        if (ZI->use_empty())
          ZI->eraseFromParent();
        return Result;
      }
    }

    return Builder.CreateIsNotNull(V, "tobool");
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  Value *Visit(Expr *E) {
    ApplyDebugLocation DL(CGF, E->getLocStart());
    return StmtVisitor<ScalarExprEmitter, Value*>::Visit(E);
  }

  Value *VisitStmt(Stmt *S) {
    S->dump(CGF.getContext().getSourceManager());
    llvm_unreachable("Stmt can't have complex result type!");
  }
  Value *VisitExpr(Expr *S);

  Value *VisitParenExpr(ParenExpr *PE) {
    return Visit(PE->getSubExpr());
  }
  Value *VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    return Visit(E->getReplacement());
  }
  Value *VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    return Visit(GE->getResultExpr());
  }

  // Leaves.
  Value *VisitIntegerLiteral(const IntegerLiteral *E) {
    return Builder.getInt(E->getValue());
  }
  Value *VisitFloatingLiteral(const FloatingLiteral *E) {
    return llvm::ConstantFP::get(VMContext, E->getValue());
  }
  Value *VisitCharacterLiteral(const CharacterLiteral *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    return EmitNullValue(E->getType());
  }
  Value *VisitGNUNullExpr(const GNUNullExpr *E) {
    return EmitNullValue(E->getType());
  }
  Value *VisitOffsetOfExpr(OffsetOfExpr *E);
  Value *VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);
  Value *VisitAddrLabelExpr(const AddrLabelExpr *E) {
    llvm::Value *V = CGF.GetAddrOfLabel(E->getLabel());
    return Builder.CreateBitCast(V, ConvertType(E->getType()));
  }

  Value *VisitSizeOfPackExpr(SizeOfPackExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()),E->getPackLength());
  }

  Value *VisitPseudoObjectExpr(PseudoObjectExpr *E) {
    return CGF.EmitPseudoObjectRValue(E).getScalarVal();
  }

  Value *VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    if (E->isGLValue())
      return EmitLoadOfLValue(CGF.getOpaqueLValueMapping(E), E->getExprLoc());

    // Otherwise, assume the mapping is the scalar directly.
    return CGF.getOpaqueRValueMapping(E).getScalarVal();
  }

  // l-values.
  Value *VisitDeclRefExpr(DeclRefExpr *E) {
    if (CodeGenFunction::ConstantEmission result = CGF.tryEmitAsConstant(E)) {
      if (result.isReference())
        return EmitLoadOfLValue(result.getReferenceLValue(CGF, E),
                                E->getExprLoc());
      return result.getValue();
    }
    return EmitLoadOfLValue(E);
  }

  Value *VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
    return CGF.EmitObjCSelectorExpr(E);
  }
  Value *VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
    return CGF.EmitObjCProtocolExpr(E);
  }
  Value *VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    return EmitLoadOfLValue(E);
  }
  Value *VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (E->getMethodDecl() &&
        E->getMethodDecl()->getReturnType()->isReferenceType())
      return EmitLoadOfLValue(E);
    return CGF.EmitObjCMessageExpr(E).getScalarVal();
  }

  Value *VisitObjCIsaExpr(ObjCIsaExpr *E) {
    LValue LV = CGF.EmitObjCIsaExpr(E);
    Value *V = CGF.EmitLoadOfLValue(LV, E->getExprLoc()).getScalarVal();
    return V;
  }

  Value *VisitArraySubscriptExpr(ArraySubscriptExpr *E);
  Value *VisitShuffleVectorExpr(ShuffleVectorExpr *E);
  Value *VisitConvertVectorExpr(ConvertVectorExpr *E);
  Value *VisitMemberExpr(MemberExpr *E);
  Value *VisitExtVectorElementExpr(Expr *E) { return EmitLoadOfLValue(E); }
  Value *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return EmitLoadOfLValue(E);
  }

  Value *VisitInitListExpr(InitListExpr *E);

  Value *VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return EmitNullValue(E->getType());
  }
  Value *VisitExplicitCastExpr(ExplicitCastExpr *E) {
    if (E->getType()->isVariablyModifiedType())
      CGF.EmitVariablyModifiedType(E->getType());

    if (CGDebugInfo *DI = CGF.getDebugInfo())
      DI->EmitExplicitCastType(E->getType());

    return VisitCastExpr(E);
  }
  Value *VisitCastExpr(CastExpr *E);

  Value *VisitCallExpr(const CallExpr *E) {
    if (E->getCallReturnType()->isReferenceType())
      return EmitLoadOfLValue(E);

    Value *V = CGF.EmitCallExpr(E).getScalarVal();

    EmitLValueAlignmentAssumption(E, V);
    return V;
  }

  Value *VisitStmtExpr(const StmtExpr *E);

  // Unary Operators.
  Value *VisitUnaryPostDec(const UnaryOperator *E) {
    LValue LV = EmitLValue(E->getSubExpr());
    return EmitScalarPrePostIncDec(E, LV, false, false);
  }
  Value *VisitUnaryPostInc(const UnaryOperator *E) {
    LValue LV = EmitLValue(E->getSubExpr());
    return EmitScalarPrePostIncDec(E, LV, true, false);
  }
  Value *VisitUnaryPreDec(const UnaryOperator *E) {
    LValue LV = EmitLValue(E->getSubExpr());
    return EmitScalarPrePostIncDec(E, LV, false, true);
  }
  Value *VisitUnaryPreInc(const UnaryOperator *E) {
    LValue LV = EmitLValue(E->getSubExpr());
    return EmitScalarPrePostIncDec(E, LV, true, true);
  }

  llvm::Value *EmitAddConsiderOverflowBehavior(const UnaryOperator *E,
                                               llvm::Value *InVal,
                                               llvm::Value *NextVal,
                                               bool IsInc);

  llvm::Value *EmitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                                       bool isInc, bool isPre);


  Value *VisitUnaryAddrOf(const UnaryOperator *E) {
    if (isa<MemberPointerType>(E->getType())) // never sugared
      return CGF.CGM.getMemberPointerConstant(E);

    return EmitLValue(E->getSubExpr()).getAddress();
  }
  Value *VisitUnaryDeref(const UnaryOperator *E) {
    if (E->getType()->isVoidType())
      return Visit(E->getSubExpr()); // the actual value should be unused
    return EmitLoadOfLValue(E);
  }
  Value *VisitUnaryPlus(const UnaryOperator *E) {
    // This differs from gcc, though, most likely due to a bug in gcc.
    TestAndClearIgnoreResultAssign();
    return Visit(E->getSubExpr());
  }
  Value *VisitUnaryMinus    (const UnaryOperator *E);
  Value *VisitUnaryNot      (const UnaryOperator *E);
  Value *VisitUnaryLNot     (const UnaryOperator *E);
  Value *VisitUnaryReal     (const UnaryOperator *E);
  Value *VisitUnaryImag     (const UnaryOperator *E);
  Value *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }

  // C++
  Value *VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E) {
    return EmitLoadOfLValue(E);
  }

  Value *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }
  Value *VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    CodeGenFunction::CXXDefaultInitExprScope Scope(CGF);
    return Visit(DIE->getExpr());
  }
  Value *VisitCXXThisExpr(CXXThisExpr *TE) {
    return CGF.LoadCXXThis();
  }

  Value *VisitExprWithCleanups(ExprWithCleanups *E) {
    CGF.enterFullExpression(E);
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    return Visit(E->getSubExpr());
  }
  Value *VisitCXXNewExpr(const CXXNewExpr *E) {
    return CGF.EmitCXXNewExpr(E);
  }
  Value *VisitCXXDeleteExpr(const CXXDeleteExpr *E) {
    CGF.EmitCXXDeleteExpr(E);
    return nullptr;
  }

  Value *VisitTypeTraitExpr(const TypeTraitExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }

  Value *VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
    return llvm::ConstantInt::get(Builder.getInt32Ty(), E->getValue());
  }

  Value *VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
    return llvm::ConstantInt::get(Builder.getInt1Ty(), E->getValue());
  }

  Value *VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *E) {
    // C++ [expr.pseudo]p1:
    //   The result shall only be used as the operand for the function call
    //   operator (), and the result of such a call has type void. The only
    //   effect is the evaluation of the postfix-expression before the dot or
    //   arrow.
    CGF.EmitScalarExpr(E->getBase());
    return nullptr;
  }

  Value *VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E) {
    return EmitNullValue(E->getType());
  }

  Value *VisitCXXThrowExpr(const CXXThrowExpr *E) {
    CGF.EmitCXXThrowExpr(E);
    return nullptr;
  }

  Value *VisitCXXNoexceptExpr(const CXXNoexceptExpr *E) {
    return Builder.getInt1(E->getValue());
  }

  // Binary Operators.
  Value *EmitMul(const BinOpInfo &Ops) {
    if (Ops.Ty->isSignedIntegerOrEnumerationType()) {
      switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
      case LangOptions::SOB_Defined:
        return Builder.CreateMul(Ops.LHS, Ops.RHS, "mul");
      case LangOptions::SOB_Undefined:
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.CreateNSWMul(Ops.LHS, Ops.RHS, "mul");
        // Fall through.
      case LangOptions::SOB_Trapping:
        return EmitOverflowCheckedBinOp(Ops);
      }
    }

    if (Ops.Ty->isUnsignedIntegerType() &&
        CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow))
      return EmitOverflowCheckedBinOp(Ops);

    if (Ops.LHS->getType()->isFPOrFPVectorTy())
      return Builder.CreateFMul(Ops.LHS, Ops.RHS, "mul");
    return Builder.CreateMul(Ops.LHS, Ops.RHS, "mul");
  }
  /// Create a binary op that checks for overflow.
  /// Currently only supports +, - and *.
  Value *EmitOverflowCheckedBinOp(const BinOpInfo &Ops);

  // Check for undefined division and modulus behaviors.
  void EmitUndefinedBehaviorIntegerDivAndRemCheck(const BinOpInfo &Ops,
                                                  llvm::Value *Zero,bool isDiv);
  // Common helper for getting how wide LHS of shift is.
  static Value *GetWidthMinusOneValue(Value* LHS,Value* RHS);
  Value *EmitDiv(const BinOpInfo &Ops);
  Value *EmitRem(const BinOpInfo &Ops);
  Value *EmitAdd(const BinOpInfo &Ops);
  Value *EmitSub(const BinOpInfo &Ops);
  Value *EmitShl(const BinOpInfo &Ops);
  Value *EmitShr(const BinOpInfo &Ops);
  Value *EmitAnd(const BinOpInfo &Ops) {
    return Builder.CreateAnd(Ops.LHS, Ops.RHS, "and");
  }
  Value *EmitXor(const BinOpInfo &Ops) {
    return Builder.CreateXor(Ops.LHS, Ops.RHS, "xor");
  }
  Value *EmitOr (const BinOpInfo &Ops) {
    return Builder.CreateOr(Ops.LHS, Ops.RHS, "or");
  }

  BinOpInfo EmitBinOps(const BinaryOperator *E);
  LValue EmitCompoundAssignLValue(const CompoundAssignOperator *E,
                            Value *(ScalarExprEmitter::*F)(const BinOpInfo &),
                                  Value *&Result);

  Value *EmitCompoundAssign(const CompoundAssignOperator *E,
                            Value *(ScalarExprEmitter::*F)(const BinOpInfo &));

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP) \
  Value *VisitBin ## OP(const BinaryOperator *E) {                         \
    return Emit ## OP(EmitBinOps(E));                                      \
  }                                                                        \
  Value *VisitBin ## OP ## Assign(const CompoundAssignOperator *E) {       \
    return EmitCompoundAssign(E, &ScalarExprEmitter::Emit ## OP);          \
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

  // Comparisons.
  Value *EmitCompare(const BinaryOperator *E, unsigned UICmpOpc,
                     unsigned SICmpOpc, unsigned FCmpOpc);
#define VISITCOMP(CODE, UI, SI, FP) \
    Value *VisitBin##CODE(const BinaryOperator *E) { \
      return EmitCompare(E, llvm::ICmpInst::UI, llvm::ICmpInst::SI, \
                         llvm::FCmpInst::FP); }
  VISITCOMP(LT, ICMP_ULT, ICMP_SLT, FCMP_OLT)
  VISITCOMP(GT, ICMP_UGT, ICMP_SGT, FCMP_OGT)
  VISITCOMP(LE, ICMP_ULE, ICMP_SLE, FCMP_OLE)
  VISITCOMP(GE, ICMP_UGE, ICMP_SGE, FCMP_OGE)
  VISITCOMP(EQ, ICMP_EQ , ICMP_EQ , FCMP_OEQ)
  VISITCOMP(NE, ICMP_NE , ICMP_NE , FCMP_UNE)
#undef VISITCOMP

  Value *VisitBinAssign     (const BinaryOperator *E);

  Value *VisitBinLAnd       (const BinaryOperator *E);
  Value *VisitBinLOr        (const BinaryOperator *E);
  Value *VisitBinComma      (const BinaryOperator *E);

  Value *VisitBinPtrMemD(const Expr *E) { return EmitLoadOfLValue(E); }
  Value *VisitBinPtrMemI(const Expr *E) { return EmitLoadOfLValue(E); }

  // Other Operators.
  Value *VisitBlockExpr(const BlockExpr *BE);
  Value *VisitAbstractConditionalOperator(const AbstractConditionalOperator *);
  Value *VisitChooseExpr(ChooseExpr *CE);
  Value *VisitVAArgExpr(VAArgExpr *VE);
  Value *VisitObjCStringLiteral(const ObjCStringLiteral *E) {
    return CGF.EmitObjCStringLiteral(E);
  }
  Value *VisitObjCBoxedExpr(ObjCBoxedExpr *E) {
    return CGF.EmitObjCBoxedExpr(E);
  }
  Value *VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
    return CGF.EmitObjCArrayLiteral(E);
  }
  Value *VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
    return CGF.EmitObjCDictionaryLiteral(E);
  }
  Value *VisitAsTypeExpr(AsTypeExpr *CE);
  Value *VisitAtomicExpr(AtomicExpr *AE);
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitConversionToBool - Convert the specified expression value to a
/// boolean (i1) truth value.  This is equivalent to "Val != 0".
Value *ScalarExprEmitter::EmitConversionToBool(Value *Src, QualType SrcType) {
  assert(SrcType.isCanonical() && "EmitScalarConversion strips typedefs");

  if (SrcType->isRealFloatingType())
    return EmitFloatToBoolConversion(Src);

  if (const MemberPointerType *MPT = dyn_cast<MemberPointerType>(SrcType))
    return CGF.CGM.getCXXABI().EmitMemberPointerIsNotNull(CGF, Src, MPT);

  assert((SrcType->isIntegerType() || isa<llvm::PointerType>(Src->getType())) &&
         "Unknown scalar type to convert");

  if (isa<llvm::IntegerType>(Src->getType()))
    return EmitIntToBoolConversion(Src);

  assert(isa<llvm::PointerType>(Src->getType()));
  return EmitPointerToBoolConversion(Src);
}

void ScalarExprEmitter::EmitFloatConversionCheck(Value *OrigSrc,
                                                 QualType OrigSrcType,
                                                 Value *Src, QualType SrcType,
                                                 QualType DstType,
                                                 llvm::Type *DstTy) {
  CodeGenFunction::SanitizerScope SanScope(&CGF);
  using llvm::APFloat;
  using llvm::APSInt;

  llvm::Type *SrcTy = Src->getType();

  llvm::Value *Check = nullptr;
  if (llvm::IntegerType *IntTy = dyn_cast<llvm::IntegerType>(SrcTy)) {
    // Integer to floating-point. This can fail for unsigned short -> __half
    // or unsigned __int128 -> float.
    assert(DstType->isFloatingType());
    bool SrcIsUnsigned = OrigSrcType->isUnsignedIntegerOrEnumerationType();

    APFloat LargestFloat =
      APFloat::getLargest(CGF.getContext().getFloatTypeSemantics(DstType));
    APSInt LargestInt(IntTy->getBitWidth(), SrcIsUnsigned);

    bool IsExact;
    if (LargestFloat.convertToInteger(LargestInt, APFloat::rmTowardZero,
                                      &IsExact) != APFloat::opOK)
      // The range of representable values of this floating point type includes
      // all values of this integer type. Don't need an overflow check.
      return;

    llvm::Value *Max = llvm::ConstantInt::get(VMContext, LargestInt);
    if (SrcIsUnsigned)
      Check = Builder.CreateICmpULE(Src, Max);
    else {
      llvm::Value *Min = llvm::ConstantInt::get(VMContext, -LargestInt);
      llvm::Value *GE = Builder.CreateICmpSGE(Src, Min);
      llvm::Value *LE = Builder.CreateICmpSLE(Src, Max);
      Check = Builder.CreateAnd(GE, LE);
    }
  } else {
    const llvm::fltSemantics &SrcSema =
      CGF.getContext().getFloatTypeSemantics(OrigSrcType);
    if (isa<llvm::IntegerType>(DstTy)) {
      // Floating-point to integer. This has undefined behavior if the source is
      // +-Inf, NaN, or doesn't fit into the destination type (after truncation
      // to an integer).
      unsigned Width = CGF.getContext().getIntWidth(DstType);
      bool Unsigned = DstType->isUnsignedIntegerOrEnumerationType();

      APSInt Min = APSInt::getMinValue(Width, Unsigned);
      APFloat MinSrc(SrcSema, APFloat::uninitialized);
      if (MinSrc.convertFromAPInt(Min, !Unsigned, APFloat::rmTowardZero) &
          APFloat::opOverflow)
        // Don't need an overflow check for lower bound. Just check for
        // -Inf/NaN.
        MinSrc = APFloat::getInf(SrcSema, true);
      else
        // Find the largest value which is too small to represent (before
        // truncation toward zero).
        MinSrc.subtract(APFloat(SrcSema, 1), APFloat::rmTowardNegative);

      APSInt Max = APSInt::getMaxValue(Width, Unsigned);
      APFloat MaxSrc(SrcSema, APFloat::uninitialized);
      if (MaxSrc.convertFromAPInt(Max, !Unsigned, APFloat::rmTowardZero) &
          APFloat::opOverflow)
        // Don't need an overflow check for upper bound. Just check for
        // +Inf/NaN.
        MaxSrc = APFloat::getInf(SrcSema, false);
      else
        // Find the smallest value which is too large to represent (before
        // truncation toward zero).
        MaxSrc.add(APFloat(SrcSema, 1), APFloat::rmTowardPositive);

      // If we're converting from __half, convert the range to float to match
      // the type of src.
      if (OrigSrcType->isHalfType()) {
        const llvm::fltSemantics &Sema =
          CGF.getContext().getFloatTypeSemantics(SrcType);
        bool IsInexact;
        MinSrc.convert(Sema, APFloat::rmTowardZero, &IsInexact);
        MaxSrc.convert(Sema, APFloat::rmTowardZero, &IsInexact);
      }

      llvm::Value *GE =
        Builder.CreateFCmpOGT(Src, llvm::ConstantFP::get(VMContext, MinSrc));
      llvm::Value *LE =
        Builder.CreateFCmpOLT(Src, llvm::ConstantFP::get(VMContext, MaxSrc));
      Check = Builder.CreateAnd(GE, LE);
    } else {
      // FIXME: Maybe split this sanitizer out from float-cast-overflow.
      //
      // Floating-point to floating-point. This has undefined behavior if the
      // source is not in the range of representable values of the destination
      // type. The C and C++ standards are spectacularly unclear here. We
      // diagnose finite out-of-range conversions, but allow infinities and NaNs
      // to convert to the corresponding value in the smaller type.
      //
      // C11 Annex F gives all such conversions defined behavior for IEC 60559
      // conforming implementations. Unfortunately, LLVM's fptrunc instruction
      // does not.

      // Converting from a lower rank to a higher rank can never have
      // undefined behavior, since higher-rank types must have a superset
      // of values of lower-rank types.
      if (CGF.getContext().getFloatingTypeOrder(OrigSrcType, DstType) != 1)
        return;

      assert(!OrigSrcType->isHalfType() &&
             "should not check conversion from __half, it has the lowest rank");

      const llvm::fltSemantics &DstSema =
        CGF.getContext().getFloatTypeSemantics(DstType);
      APFloat MinBad = APFloat::getLargest(DstSema, false);
      APFloat MaxBad = APFloat::getInf(DstSema, false);

      bool IsInexact;
      MinBad.convert(SrcSema, APFloat::rmTowardZero, &IsInexact);
      MaxBad.convert(SrcSema, APFloat::rmTowardZero, &IsInexact);

      Value *AbsSrc = CGF.EmitNounwindRuntimeCall(
        CGF.CGM.getIntrinsic(llvm::Intrinsic::fabs, Src->getType()), Src);
      llvm::Value *GE =
        Builder.CreateFCmpOGT(AbsSrc, llvm::ConstantFP::get(VMContext, MinBad));
      llvm::Value *LE =
        Builder.CreateFCmpOLT(AbsSrc, llvm::ConstantFP::get(VMContext, MaxBad));
      Check = Builder.CreateNot(Builder.CreateAnd(GE, LE));
    }
  }

  // FIXME: Provide a SourceLocation.
  llvm::Constant *StaticArgs[] = {
    CGF.EmitCheckTypeDescriptor(OrigSrcType),
    CGF.EmitCheckTypeDescriptor(DstType)
  };
  CGF.EmitCheck(std::make_pair(Check, SanitizerKind::FloatCastOverflow),
                "float_cast_overflow", StaticArgs, OrigSrc);
}

/// EmitScalarConversion - Emit a conversion from the specified type to the
/// specified destination type, both of which are LLVM scalar types.
Value *ScalarExprEmitter::EmitScalarConversion(Value *Src, QualType SrcType,
                                               QualType DstType) {
  SrcType = CGF.getContext().getCanonicalType(SrcType);
  DstType = CGF.getContext().getCanonicalType(DstType);
  if (SrcType == DstType) return Src;

  if (DstType->isVoidType()) return nullptr;

  llvm::Value *OrigSrc = Src;
  QualType OrigSrcType = SrcType;
  llvm::Type *SrcTy = Src->getType();

  // If casting to/from storage-only half FP, use special intrinsics.
  if (SrcType->isHalfType() && !CGF.getContext().getLangOpts().NativeHalfType &&
      !CGF.getContext().getLangOpts().HalfArgsAndReturns) {
    Src = Builder.CreateCall(
        CGF.CGM.getIntrinsic(llvm::Intrinsic::convert_from_fp16,
                             CGF.CGM.FloatTy),
        Src);
    SrcType = CGF.getContext().FloatTy;
    SrcTy = CGF.FloatTy;
  }

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (DstType->isBooleanType())
    return EmitConversionToBool(Src, SrcType);

  llvm::Type *DstTy = ConvertType(DstType);

  // Ignore conversions like int -> uint.
  if (SrcTy == DstTy)
    return Src;

  // Handle pointer conversions next: pointers can only be converted to/from
  // other pointers and integers. Check for pointer types in terms of LLVM, as
  // some native types (like Obj-C id) may map to a pointer type.
  if (isa<llvm::PointerType>(DstTy)) {
    // The source value may be an integer, or a pointer.
    if (isa<llvm::PointerType>(SrcTy))
      return Builder.CreateBitCast(Src, DstTy, "conv");

    assert(SrcType->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
    // First, convert to the correct width so that we control the kind of
    // extension.
    llvm::Type *MiddleTy = CGF.IntPtrTy;
    bool InputSigned = SrcType->isSignedIntegerOrEnumerationType();
    llvm::Value* IntResult =
        Builder.CreateIntCast(Src, MiddleTy, InputSigned, "conv");
    // Then, cast to pointer.
    return Builder.CreateIntToPtr(IntResult, DstTy, "conv");
  }

  if (isa<llvm::PointerType>(SrcTy)) {
    // Must be an ptr to int cast.
    assert(isa<llvm::IntegerType>(DstTy) && "not ptr->int?");
    return Builder.CreatePtrToInt(Src, DstTy, "conv");
  }

  // A scalar can be splatted to an extended vector of the same element type
  if (DstType->isExtVectorType() && !SrcType->isVectorType()) {
    // Cast the scalar to element type
    QualType EltTy = DstType->getAs<ExtVectorType>()->getElementType();
    llvm::Value *Elt = EmitScalarConversion(Src, SrcType, EltTy);

    // Splat the element across to all elements
    unsigned NumElements = cast<llvm::VectorType>(DstTy)->getNumElements();
    return Builder.CreateVectorSplat(NumElements, Elt, "splat");
  }

  // Allow bitcast from vector to integer/fp of the same size.
  if (isa<llvm::VectorType>(SrcTy) ||
      isa<llvm::VectorType>(DstTy))
    return Builder.CreateBitCast(Src, DstTy, "conv");

  // Finally, we have the arithmetic types: real int/float.
  Value *Res = nullptr;
  llvm::Type *ResTy = DstTy;

  // An overflowing conversion has undefined behavior if either the source type
  // or the destination type is a floating-point type.
  if (CGF.SanOpts.has(SanitizerKind::FloatCastOverflow) &&
      (OrigSrcType->isFloatingType() || DstType->isFloatingType()))
    EmitFloatConversionCheck(OrigSrc, OrigSrcType, Src, SrcType, DstType,
                             DstTy);

  // Cast to half via float
  if (DstType->isHalfType() && !CGF.getContext().getLangOpts().NativeHalfType &&
      !CGF.getContext().getLangOpts().HalfArgsAndReturns)
    DstTy = CGF.FloatTy;

  if (isa<llvm::IntegerType>(SrcTy)) {
    bool InputSigned = SrcType->isSignedIntegerOrEnumerationType();
    if (isa<llvm::IntegerType>(DstTy))
      Res = Builder.CreateIntCast(Src, DstTy, InputSigned, "conv");
    else if (InputSigned)
      Res = Builder.CreateSIToFP(Src, DstTy, "conv");
    else
      Res = Builder.CreateUIToFP(Src, DstTy, "conv");
  } else if (isa<llvm::IntegerType>(DstTy)) {
    assert(SrcTy->isFloatingPointTy() && "Unknown real conversion");
    if (DstType->isSignedIntegerOrEnumerationType())
      Res = Builder.CreateFPToSI(Src, DstTy, "conv");
    else
      Res = Builder.CreateFPToUI(Src, DstTy, "conv");
  } else {
    assert(SrcTy->isFloatingPointTy() && DstTy->isFloatingPointTy() &&
           "Unknown real conversion");
    if (DstTy->getTypeID() < SrcTy->getTypeID())
      Res = Builder.CreateFPTrunc(Src, DstTy, "conv");
    else
      Res = Builder.CreateFPExt(Src, DstTy, "conv");
  }

  if (DstTy != ResTy) {
    assert(ResTy->isIntegerTy(16) && "Only half FP requires extra conversion");
    Res = Builder.CreateCall(
        CGF.CGM.getIntrinsic(llvm::Intrinsic::convert_to_fp16, CGF.CGM.FloatTy),
        Res);
  }

  return Res;
}

/// EmitComplexToScalarConversion - Emit a conversion from the specified complex
/// type to the specified destination type, where the destination type is an
/// LLVM scalar type.
Value *ScalarExprEmitter::
EmitComplexToScalarConversion(CodeGenFunction::ComplexPairTy Src,
                              QualType SrcTy, QualType DstTy) {
  // Get the source element type.
  SrcTy = SrcTy->castAs<ComplexType>()->getElementType();

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (DstTy->isBooleanType()) {
    //  Complex != 0  -> (Real != 0) | (Imag != 0)
    Src.first  = EmitScalarConversion(Src.first, SrcTy, DstTy);
    Src.second = EmitScalarConversion(Src.second, SrcTy, DstTy);
    return Builder.CreateOr(Src.first, Src.second, "tobool");
  }

  // C99 6.3.1.7p2: "When a value of complex type is converted to a real type,
  // the imaginary part of the complex value is discarded and the value of the
  // real part is converted according to the conversion rules for the
  // corresponding real type.
  return EmitScalarConversion(Src.first, SrcTy, DstTy);
}

Value *ScalarExprEmitter::EmitNullValue(QualType Ty) {
  return CGF.EmitFromMemory(CGF.CGM.EmitNullConstant(Ty), Ty);
}

/// \brief Emit a sanitization check for the given "binary" operation (which
/// might actually be a unary increment which has been lowered to a binary
/// operation). The check passes if all values in \p Checks (which are \c i1),
/// are \c true.
void ScalarExprEmitter::EmitBinOpCheck(
    ArrayRef<std::pair<Value *, SanitizerKind>> Checks, const BinOpInfo &Info) {
  assert(CGF.IsSanitizerScope);
  StringRef CheckName;
  SmallVector<llvm::Constant *, 4> StaticData;
  SmallVector<llvm::Value *, 2> DynamicData;

  BinaryOperatorKind Opcode = Info.Opcode;
  if (BinaryOperator::isCompoundAssignmentOp(Opcode))
    Opcode = BinaryOperator::getOpForCompoundAssignment(Opcode);

  StaticData.push_back(CGF.EmitCheckSourceLocation(Info.E->getExprLoc()));
  const UnaryOperator *UO = dyn_cast<UnaryOperator>(Info.E);
  if (UO && UO->getOpcode() == UO_Minus) {
    CheckName = "negate_overflow";
    StaticData.push_back(CGF.EmitCheckTypeDescriptor(UO->getType()));
    DynamicData.push_back(Info.RHS);
  } else {
    if (BinaryOperator::isShiftOp(Opcode)) {
      // Shift LHS negative or too large, or RHS out of bounds.
      CheckName = "shift_out_of_bounds";
      const BinaryOperator *BO = cast<BinaryOperator>(Info.E);
      StaticData.push_back(
        CGF.EmitCheckTypeDescriptor(BO->getLHS()->getType()));
      StaticData.push_back(
        CGF.EmitCheckTypeDescriptor(BO->getRHS()->getType()));
    } else if (Opcode == BO_Div || Opcode == BO_Rem) {
      // Divide or modulo by zero, or signed overflow (eg INT_MAX / -1).
      CheckName = "divrem_overflow";
      StaticData.push_back(CGF.EmitCheckTypeDescriptor(Info.Ty));
    } else {
      // Arithmetic overflow (+, -, *).
      switch (Opcode) {
      case BO_Add: CheckName = "add_overflow"; break;
      case BO_Sub: CheckName = "sub_overflow"; break;
      case BO_Mul: CheckName = "mul_overflow"; break;
      default: llvm_unreachable("unexpected opcode for bin op check");
      }
      StaticData.push_back(CGF.EmitCheckTypeDescriptor(Info.Ty));
    }
    DynamicData.push_back(Info.LHS);
    DynamicData.push_back(Info.RHS);
  }

  CGF.EmitCheck(Checks, CheckName, StaticData, DynamicData);
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

Value *ScalarExprEmitter::VisitExpr(Expr *E) {
  CGF.ErrorUnsupported(E, "scalar expression");
  if (E->getType()->isVoidType())
    return nullptr;
  return llvm::UndefValue::get(CGF.ConvertType(E->getType()));
}

Value *ScalarExprEmitter::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  // Vector Mask Case
  if (E->getNumSubExprs() == 2 ||
      (E->getNumSubExprs() == 3 && E->getExpr(2)->getType()->isVectorType())) {
    Value *LHS = CGF.EmitScalarExpr(E->getExpr(0));
    Value *RHS = CGF.EmitScalarExpr(E->getExpr(1));
    Value *Mask;

    llvm::VectorType *LTy = cast<llvm::VectorType>(LHS->getType());
    unsigned LHSElts = LTy->getNumElements();

    if (E->getNumSubExprs() == 3) {
      Mask = CGF.EmitScalarExpr(E->getExpr(2));

      // Shuffle LHS & RHS into one input vector.
      SmallVector<llvm::Constant*, 32> concat;
      for (unsigned i = 0; i != LHSElts; ++i) {
        concat.push_back(Builder.getInt32(2*i));
        concat.push_back(Builder.getInt32(2*i+1));
      }

      Value* CV = llvm::ConstantVector::get(concat);
      LHS = Builder.CreateShuffleVector(LHS, RHS, CV, "concat");
      LHSElts *= 2;
    } else {
      Mask = RHS;
    }

    llvm::VectorType *MTy = cast<llvm::VectorType>(Mask->getType());
    llvm::Constant* EltMask;

    EltMask = llvm::ConstantInt::get(MTy->getElementType(),
                                     llvm::NextPowerOf2(LHSElts-1)-1);

    // Mask off the high bits of each shuffle index.
    Value *MaskBits = llvm::ConstantVector::getSplat(MTy->getNumElements(),
                                                     EltMask);
    Mask = Builder.CreateAnd(Mask, MaskBits, "mask");

    // newv = undef
    // mask = mask & maskbits
    // for each elt
    //   n = extract mask i
    //   x = extract val n
    //   newv = insert newv, x, i
    llvm::VectorType *RTy = llvm::VectorType::get(LTy->getElementType(),
                                                  MTy->getNumElements());
    Value* NewV = llvm::UndefValue::get(RTy);
    for (unsigned i = 0, e = MTy->getNumElements(); i != e; ++i) {
      Value *IIndx = llvm::ConstantInt::get(CGF.SizeTy, i);
      Value *Indx = Builder.CreateExtractElement(Mask, IIndx, "shuf_idx");

      Value *VExt = Builder.CreateExtractElement(LHS, Indx, "shuf_elt");
      NewV = Builder.CreateInsertElement(NewV, VExt, IIndx, "shuf_ins");
    }
    return NewV;
  }

  Value* V1 = CGF.EmitScalarExpr(E->getExpr(0));
  Value* V2 = CGF.EmitScalarExpr(E->getExpr(1));

  SmallVector<llvm::Constant*, 32> indices;
  for (unsigned i = 2; i < E->getNumSubExprs(); ++i) {
    llvm::APSInt Idx = E->getShuffleMaskIdx(CGF.getContext(), i-2);
    // Check for -1 and output it as undef in the IR.
    if (Idx.isSigned() && Idx.isAllOnesValue())
      indices.push_back(llvm::UndefValue::get(CGF.Int32Ty));
    else
      indices.push_back(Builder.getInt32(Idx.getZExtValue()));
  }

  Value *SV = llvm::ConstantVector::get(indices);
  return Builder.CreateShuffleVector(V1, V2, SV, "shuffle");
}

Value *ScalarExprEmitter::VisitConvertVectorExpr(ConvertVectorExpr *E) {
  QualType SrcType = E->getSrcExpr()->getType(),
           DstType = E->getType();

  Value *Src  = CGF.EmitScalarExpr(E->getSrcExpr());

  SrcType = CGF.getContext().getCanonicalType(SrcType);
  DstType = CGF.getContext().getCanonicalType(DstType);
  if (SrcType == DstType) return Src;

  assert(SrcType->isVectorType() &&
         "ConvertVector source type must be a vector");
  assert(DstType->isVectorType() &&
         "ConvertVector destination type must be a vector");

  llvm::Type *SrcTy = Src->getType();
  llvm::Type *DstTy = ConvertType(DstType);

  // Ignore conversions like int -> uint.
  if (SrcTy == DstTy)
    return Src;

  QualType SrcEltType = SrcType->getAs<VectorType>()->getElementType(),
           DstEltType = DstType->getAs<VectorType>()->getElementType();

  assert(SrcTy->isVectorTy() &&
         "ConvertVector source IR type must be a vector");
  assert(DstTy->isVectorTy() &&
         "ConvertVector destination IR type must be a vector");

  llvm::Type *SrcEltTy = SrcTy->getVectorElementType(),
             *DstEltTy = DstTy->getVectorElementType();

  if (DstEltType->isBooleanType()) {
    assert((SrcEltTy->isFloatingPointTy() ||
            isa<llvm::IntegerType>(SrcEltTy)) && "Unknown boolean conversion");

    llvm::Value *Zero = llvm::Constant::getNullValue(SrcTy);
    if (SrcEltTy->isFloatingPointTy()) {
      return Builder.CreateFCmpUNE(Src, Zero, "tobool");
    } else {
      return Builder.CreateICmpNE(Src, Zero, "tobool");
    }
  }

  // We have the arithmetic types: real int/float.
  Value *Res = nullptr;

  if (isa<llvm::IntegerType>(SrcEltTy)) {
    bool InputSigned = SrcEltType->isSignedIntegerOrEnumerationType();
    if (isa<llvm::IntegerType>(DstEltTy))
      Res = Builder.CreateIntCast(Src, DstTy, InputSigned, "conv");
    else if (InputSigned)
      Res = Builder.CreateSIToFP(Src, DstTy, "conv");
    else
      Res = Builder.CreateUIToFP(Src, DstTy, "conv");
  } else if (isa<llvm::IntegerType>(DstEltTy)) {
    assert(SrcEltTy->isFloatingPointTy() && "Unknown real conversion");
    if (DstEltType->isSignedIntegerOrEnumerationType())
      Res = Builder.CreateFPToSI(Src, DstTy, "conv");
    else
      Res = Builder.CreateFPToUI(Src, DstTy, "conv");
  } else {
    assert(SrcEltTy->isFloatingPointTy() && DstEltTy->isFloatingPointTy() &&
           "Unknown real conversion");
    if (DstEltTy->getTypeID() < SrcEltTy->getTypeID())
      Res = Builder.CreateFPTrunc(Src, DstTy, "conv");
    else
      Res = Builder.CreateFPExt(Src, DstTy, "conv");
  }

  return Res;
}

Value *ScalarExprEmitter::VisitMemberExpr(MemberExpr *E) {
  llvm::APSInt Value;
  if (E->EvaluateAsInt(Value, CGF.getContext(), Expr::SE_AllowSideEffects)) {
    if (E->isArrow())
      CGF.EmitScalarExpr(E->getBase());
    else
      EmitLValue(E->getBase());
    return Builder.getInt(Value);
  }

  return EmitLoadOfLValue(E);
}

Value *ScalarExprEmitter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  TestAndClearIgnoreResultAssign();

  // Emit subscript expressions in rvalue context's.  For most cases, this just
  // loads the lvalue formed by the subscript expr.  However, we have to be
  // careful, because the base of a vector subscript is occasionally an rvalue,
  // so we can't get it as an lvalue.
  if (!E->getBase()->getType()->isVectorType())
    return EmitLoadOfLValue(E);

  // Handle the vector case.  The base must be a vector, the index must be an
  // integer value.
  Value *Base = Visit(E->getBase());
  Value *Idx  = Visit(E->getIdx());
  QualType IdxTy = E->getIdx()->getType();

  if (CGF.SanOpts.has(SanitizerKind::ArrayBounds))
    CGF.EmitBoundsCheck(E, E->getBase(), Idx, IdxTy, /*Accessed*/true);

  return Builder.CreateExtractElement(Base, Idx, "vecext");
}

static llvm::Constant *getMaskElt(llvm::ShuffleVectorInst *SVI, unsigned Idx,
                                  unsigned Off, llvm::Type *I32Ty) {
  int MV = SVI->getMaskValue(Idx);
  if (MV == -1)
    return llvm::UndefValue::get(I32Ty);
  return llvm::ConstantInt::get(I32Ty, Off+MV);
}

Value *ScalarExprEmitter::VisitInitListExpr(InitListExpr *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();
  (void)Ignore;
  assert (Ignore == false && "init list ignored");
  unsigned NumInitElements = E->getNumInits();

  if (E->hadArrayRangeDesignator())
    CGF.ErrorUnsupported(E, "GNU array range designator extension");

  llvm::VectorType *VType =
    dyn_cast<llvm::VectorType>(ConvertType(E->getType()));

  if (!VType) {
    if (NumInitElements == 0) {
      // C++11 value-initialization for the scalar.
      return EmitNullValue(E->getType());
    }
    // We have a scalar in braces. Just use the first element.
    return Visit(E->getInit(0));
  }

  unsigned ResElts = VType->getNumElements();

  // Loop over initializers collecting the Value for each, and remembering
  // whether the source was swizzle (ExtVectorElementExpr).  This will allow
  // us to fold the shuffle for the swizzle into the shuffle for the vector
  // initializer, since LLVM optimizers generally do not want to touch
  // shuffles.
  unsigned CurIdx = 0;
  bool VIsUndefShuffle = false;
  llvm::Value *V = llvm::UndefValue::get(VType);
  for (unsigned i = 0; i != NumInitElements; ++i) {
    Expr *IE = E->getInit(i);
    Value *Init = Visit(IE);
    SmallVector<llvm::Constant*, 16> Args;

    llvm::VectorType *VVT = dyn_cast<llvm::VectorType>(Init->getType());

    // Handle scalar elements.  If the scalar initializer is actually one
    // element of a different vector of the same width, use shuffle instead of
    // extract+insert.
    if (!VVT) {
      if (isa<ExtVectorElementExpr>(IE)) {
        llvm::ExtractElementInst *EI = cast<llvm::ExtractElementInst>(Init);

        if (EI->getVectorOperandType()->getNumElements() == ResElts) {
          llvm::ConstantInt *C = cast<llvm::ConstantInt>(EI->getIndexOperand());
          Value *LHS = nullptr, *RHS = nullptr;
          if (CurIdx == 0) {
            // insert into undef -> shuffle (src, undef)
            Args.push_back(C);
            Args.resize(ResElts, llvm::UndefValue::get(CGF.Int32Ty));

            LHS = EI->getVectorOperand();
            RHS = V;
            VIsUndefShuffle = true;
          } else if (VIsUndefShuffle) {
            // insert into undefshuffle && size match -> shuffle (v, src)
            llvm::ShuffleVectorInst *SVV = cast<llvm::ShuffleVectorInst>(V);
            for (unsigned j = 0; j != CurIdx; ++j)
              Args.push_back(getMaskElt(SVV, j, 0, CGF.Int32Ty));
            Args.push_back(Builder.getInt32(ResElts + C->getZExtValue()));
            Args.resize(ResElts, llvm::UndefValue::get(CGF.Int32Ty));

            LHS = cast<llvm::ShuffleVectorInst>(V)->getOperand(0);
            RHS = EI->getVectorOperand();
            VIsUndefShuffle = false;
          }
          if (!Args.empty()) {
            llvm::Constant *Mask = llvm::ConstantVector::get(Args);
            V = Builder.CreateShuffleVector(LHS, RHS, Mask);
            ++CurIdx;
            continue;
          }
        }
      }
      V = Builder.CreateInsertElement(V, Init, Builder.getInt32(CurIdx),
                                      "vecinit");
      VIsUndefShuffle = false;
      ++CurIdx;
      continue;
    }

    unsigned InitElts = VVT->getNumElements();

    // If the initializer is an ExtVecEltExpr (a swizzle), and the swizzle's
    // input is the same width as the vector being constructed, generate an
    // optimized shuffle of the swizzle input into the result.
    unsigned Offset = (CurIdx == 0) ? 0 : ResElts;
    if (isa<ExtVectorElementExpr>(IE)) {
      llvm::ShuffleVectorInst *SVI = cast<llvm::ShuffleVectorInst>(Init);
      Value *SVOp = SVI->getOperand(0);
      llvm::VectorType *OpTy = cast<llvm::VectorType>(SVOp->getType());

      if (OpTy->getNumElements() == ResElts) {
        for (unsigned j = 0; j != CurIdx; ++j) {
          // If the current vector initializer is a shuffle with undef, merge
          // this shuffle directly into it.
          if (VIsUndefShuffle) {
            Args.push_back(getMaskElt(cast<llvm::ShuffleVectorInst>(V), j, 0,
                                      CGF.Int32Ty));
          } else {
            Args.push_back(Builder.getInt32(j));
          }
        }
        for (unsigned j = 0, je = InitElts; j != je; ++j)
          Args.push_back(getMaskElt(SVI, j, Offset, CGF.Int32Ty));
        Args.resize(ResElts, llvm::UndefValue::get(CGF.Int32Ty));

        if (VIsUndefShuffle)
          V = cast<llvm::ShuffleVectorInst>(V)->getOperand(0);

        Init = SVOp;
      }
    }

    // Extend init to result vector length, and then shuffle its contribution
    // to the vector initializer into V.
    if (Args.empty()) {
      for (unsigned j = 0; j != InitElts; ++j)
        Args.push_back(Builder.getInt32(j));
      Args.resize(ResElts, llvm::UndefValue::get(CGF.Int32Ty));
      llvm::Constant *Mask = llvm::ConstantVector::get(Args);
      Init = Builder.CreateShuffleVector(Init, llvm::UndefValue::get(VVT),
                                         Mask, "vext");

      Args.clear();
      for (unsigned j = 0; j != CurIdx; ++j)
        Args.push_back(Builder.getInt32(j));
      for (unsigned j = 0; j != InitElts; ++j)
        Args.push_back(Builder.getInt32(j+Offset));
      Args.resize(ResElts, llvm::UndefValue::get(CGF.Int32Ty));
    }

    // If V is undef, make sure it ends up on the RHS of the shuffle to aid
    // merging subsequent shuffles into this one.
    if (CurIdx == 0)
      std::swap(V, Init);
    llvm::Constant *Mask = llvm::ConstantVector::get(Args);
    V = Builder.CreateShuffleVector(V, Init, Mask, "vecinit");
    VIsUndefShuffle = isa<llvm::UndefValue>(Init);
    CurIdx += InitElts;
  }

  // FIXME: evaluate codegen vs. shuffling against constant null vector.
  // Emit remaining default initializers.
  llvm::Type *EltTy = VType->getElementType();

  // Emit remaining default initializers
  for (/* Do not initialize i*/; CurIdx < ResElts; ++CurIdx) {
    Value *Idx = Builder.getInt32(CurIdx);
    llvm::Value *Init = llvm::Constant::getNullValue(EltTy);
    V = Builder.CreateInsertElement(V, Init, Idx, "vecinit");
  }
  return V;
}

static bool ShouldNullCheckClassCastValue(const CastExpr *CE) {
  const Expr *E = CE->getSubExpr();

  if (CE->getCastKind() == CK_UncheckedDerivedToBase)
    return false;

  if (isa<CXXThisExpr>(E)) {
    // We always assume that 'this' is never null.
    return false;
  }

  if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(CE)) {
    // And that glvalue casts are never null.
    if (ICE->getValueKind() != VK_RValue)
      return false;
  }

  return true;
}

// VisitCastExpr - Emit code for an explicit or implicit cast.  Implicit casts
// have to handle a more broad range of conversions than explicit casts, as they
// handle things like function to ptr-to-function decay etc.
Value *ScalarExprEmitter::VisitCastExpr(CastExpr *CE) {
  Expr *E = CE->getSubExpr();
  QualType DestTy = CE->getType();
  CastKind Kind = CE->getCastKind();

  if (!DestTy->isVoidType())
    TestAndClearIgnoreResultAssign();

  // Since almost all cast kinds apply to scalars, this switch doesn't have
  // a default case, so the compiler will warn on a missing case.  The cases
  // are in the same order as in the CastKind enum.
  switch (Kind) {
  case CK_Dependent: llvm_unreachable("dependent cast kind in IR gen!");
  case CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  case CK_LValueBitCast:
  case CK_ObjCObjectLValueCast: {
    Value *V = EmitLValue(E).getAddress();
    V = Builder.CreateBitCast(V,
                          ConvertType(CGF.getContext().getPointerType(DestTy)));
    return EmitLoadOfLValue(CGF.MakeNaturalAlignAddrLValue(V, DestTy),
                            CE->getExprLoc());
  }

  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_BitCast: {
    Value *Src = Visit(const_cast<Expr*>(E));
    llvm::Type *SrcTy = Src->getType();
    llvm::Type *DstTy = ConvertType(DestTy);
    if (SrcTy->isPtrOrPtrVectorTy() && DstTy->isPtrOrPtrVectorTy() &&
        SrcTy->getPointerAddressSpace() != DstTy->getPointerAddressSpace()) {
      llvm_unreachable("wrong cast for pointers in different address spaces"
                       "(must be an address space cast)!");
    }
    return Builder.CreateBitCast(Src, DstTy);
  }
  case CK_AddressSpaceConversion: {
    Value *Src = Visit(const_cast<Expr*>(E));
    return Builder.CreateAddrSpaceCast(Src, ConvertType(DestTy));
  }
  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
    return Visit(const_cast<Expr*>(E));

  case CK_BaseToDerived: {
    const CXXRecordDecl *DerivedClassDecl = DestTy->getPointeeCXXRecordDecl();
    assert(DerivedClassDecl && "BaseToDerived arg isn't a C++ object pointer!");

    llvm::Value *V = Visit(E);

    llvm::Value *Derived =
      CGF.GetAddressOfDerivedClass(V, DerivedClassDecl,
                                   CE->path_begin(), CE->path_end(),
                                   ShouldNullCheckClassCastValue(CE));

    // C++11 [expr.static.cast]p11: Behavior is undefined if a downcast is
    // performed and the object is not of the derived type.
    if (CGF.sanitizePerformTypeCheck())
      CGF.EmitTypeCheck(CodeGenFunction::TCK_DowncastPointer, CE->getExprLoc(),
                        Derived, DestTy->getPointeeType());

    return Derived;
  }
  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    const CXXRecordDecl *DerivedClassDecl =
      E->getType()->getPointeeCXXRecordDecl();
    assert(DerivedClassDecl && "DerivedToBase arg isn't a C++ object pointer!");

    return CGF.GetAddressOfBaseClass(
        Visit(E), DerivedClassDecl, CE->path_begin(), CE->path_end(),
        ShouldNullCheckClassCastValue(CE), CE->getExprLoc());
  }
  case CK_Dynamic: {
    Value *V = Visit(const_cast<Expr*>(E));
    const CXXDynamicCastExpr *DCE = cast<CXXDynamicCastExpr>(CE);
    return CGF.EmitDynamicCast(V, DCE);
  }

  case CK_ArrayToPointerDecay: {
    assert(E->getType()->isArrayType() &&
           "Array to pointer decay must have array source type!");

    Value *V = EmitLValue(E).getAddress();  // Bitfields can't be arrays.

    // Note that VLA pointers are always decayed, so we don't need to do
    // anything here.
    if (!E->getType()->isVariableArrayType()) {
      assert(isa<llvm::PointerType>(V->getType()) && "Expected pointer");
      V = CGF.Builder.CreatePointerCast(
          V, ConvertType(E->getType())->getPointerTo(
            V->getType()->getPointerAddressSpace()));

      assert(isa<llvm::ArrayType>(V->getType()->getPointerElementType()) &&
             "Expected pointer to array");
      V = Builder.CreateStructGEP(V, 0, "arraydecay");
    }

    // Make sure the array decay ends up being the right type.  This matters if
    // the array type was of an incomplete type.
    return CGF.Builder.CreatePointerCast(V, ConvertType(CE->getType()));
  }
  case CK_FunctionToPointerDecay:
    return EmitLValue(E).getAddress();

  case CK_NullToPointer:
    if (MustVisitNullValue(E))
      (void) Visit(E);

    return llvm::ConstantPointerNull::get(
                               cast<llvm::PointerType>(ConvertType(DestTy)));

  case CK_NullToMemberPointer: {
    if (MustVisitNullValue(E))
      (void) Visit(E);

    const MemberPointerType *MPT = CE->getType()->getAs<MemberPointerType>();
    return CGF.CGM.getCXXABI().EmitNullMemberPointer(MPT);
  }

  case CK_ReinterpretMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer: {
    Value *Src = Visit(E);

    // Note that the AST doesn't distinguish between checked and
    // unchecked member pointer conversions, so we always have to
    // implement checked conversions here.  This is inefficient when
    // actual control flow may be required in order to perform the
    // check, which it is for data member pointers (but not member
    // function pointers on Itanium and ARM).
    return CGF.CGM.getCXXABI().EmitMemberPointerConversion(CGF, CE, Src);
  }

  case CK_ARCProduceObject:
    return CGF.EmitARCRetainScalarExpr(E);
  case CK_ARCConsumeObject:
    return CGF.EmitObjCConsumeObject(E->getType(), Visit(E));
  case CK_ARCReclaimReturnedObject: {
    llvm::Value *value = Visit(E);
    value = CGF.EmitARCRetainAutoreleasedReturnValue(value);
    return CGF.EmitObjCConsumeObject(E->getType(), value);
  }
  case CK_ARCExtendBlockObject:
    return CGF.EmitARCExtendBlockObject(E);

  case CK_CopyAndAutoreleaseBlockObject:
    return CGF.EmitBlockCopyAndAutorelease(Visit(E), E->getType());

  case CK_FloatingRealToComplex:
  case CK_FloatingComplexCast:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_FloatingComplexToIntegralComplex:
  case CK_ConstructorConversion:
  case CK_ToUnion:
    llvm_unreachable("scalar cast to non-scalar value");

  case CK_LValueToRValue:
    assert(CGF.getContext().hasSameUnqualifiedType(E->getType(), DestTy));
    assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr*>(E));

  case CK_IntegralToPointer: {
    Value *Src = Visit(const_cast<Expr*>(E));

    // First, convert to the correct width so that we control the kind of
    // extension.
    llvm::Type *MiddleTy = CGF.IntPtrTy;
    bool InputSigned = E->getType()->isSignedIntegerOrEnumerationType();
    llvm::Value* IntResult =
      Builder.CreateIntCast(Src, MiddleTy, InputSigned, "conv");

    return Builder.CreateIntToPtr(IntResult, ConvertType(DestTy));
  }
  case CK_PointerToIntegral:
    assert(!DestTy->isBooleanType() && "bool should use PointerToBool");
    return Builder.CreatePtrToInt(Visit(E), ConvertType(DestTy));

  case CK_ToVoid: {
    CGF.EmitIgnoredExpr(E);
    return nullptr;
  }
  case CK_VectorSplat: {
    llvm::Type *DstTy = ConvertType(DestTy);
    Value *Elt = Visit(const_cast<Expr*>(E));
    Elt = EmitScalarConversion(Elt, E->getType(),
                               DestTy->getAs<VectorType>()->getElementType());

    // Splat the element across to all elements
    unsigned NumElements = cast<llvm::VectorType>(DstTy)->getNumElements();
    return Builder.CreateVectorSplat(NumElements, Elt, "splat");
  }

  case CK_IntegralCast:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingCast:
    return EmitScalarConversion(Visit(E), E->getType(), DestTy);
  case CK_IntegralToBoolean:
    return EmitIntToBoolConversion(Visit(E));
  case CK_PointerToBoolean:
    return EmitPointerToBoolConversion(Visit(E));
  case CK_FloatingToBoolean:
    return EmitFloatToBoolConversion(Visit(E));
  case CK_MemberPointerToBoolean: {
    llvm::Value *MemPtr = Visit(E);
    const MemberPointerType *MPT = E->getType()->getAs<MemberPointerType>();
    return CGF.CGM.getCXXABI().EmitMemberPointerIsNotNull(CGF, MemPtr, MPT);
  }

  case CK_FloatingComplexToReal:
  case CK_IntegralComplexToReal:
    return CGF.EmitComplexExpr(E, false, true).first;

  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    CodeGenFunction::ComplexPairTy V = CGF.EmitComplexExpr(E);

    // TODO: kill this function off, inline appropriate case here
    return EmitComplexToScalarConversion(V, E->getType(), DestTy);
  }

  case CK_ZeroToOCLEvent: {
    assert(DestTy->isEventT() && "CK_ZeroToOCLEvent cast on non-event type");
    return llvm::Constant::getNullValue(ConvertType(DestTy));
  }

  }

  llvm_unreachable("unknown scalar cast");
}

Value *ScalarExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  CodeGenFunction::StmtExprEvaluation eval(CGF);
  llvm::Value *RetAlloca = CGF.EmitCompoundStmt(*E->getSubStmt(),
                                                !E->getType()->isVoidType());
  if (!RetAlloca)
    return nullptr;
  return CGF.EmitLoadOfScalar(CGF.MakeAddrLValue(RetAlloca, E->getType()),
                              E->getExprLoc());
}

//===----------------------------------------------------------------------===//
//                             Unary Operators
//===----------------------------------------------------------------------===//

llvm::Value *ScalarExprEmitter::
EmitAddConsiderOverflowBehavior(const UnaryOperator *E,
                                llvm::Value *InVal,
                                llvm::Value *NextVal, bool IsInc) {
  switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
  case LangOptions::SOB_Defined:
    return Builder.CreateAdd(InVal, NextVal, IsInc ? "inc" : "dec");
  case LangOptions::SOB_Undefined:
    if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
      return Builder.CreateNSWAdd(InVal, NextVal, IsInc ? "inc" : "dec");
    // Fall through.
  case LangOptions::SOB_Trapping:
    BinOpInfo BinOp;
    BinOp.LHS = InVal;
    BinOp.RHS = NextVal;
    BinOp.Ty = E->getType();
    BinOp.Opcode = BO_Add;
    BinOp.FPContractable = false;
    BinOp.E = E;
    return EmitOverflowCheckedBinOp(BinOp);
  }
  llvm_unreachable("Unknown SignedOverflowBehaviorTy");
}

llvm::Value *
ScalarExprEmitter::EmitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                                           bool isInc, bool isPre) {

  QualType type = E->getSubExpr()->getType();
  llvm::PHINode *atomicPHI = nullptr;
  llvm::Value *value;
  llvm::Value *input;

  int amount = (isInc ? 1 : -1);

  if (const AtomicType *atomicTy = type->getAs<AtomicType>()) {
    type = atomicTy->getValueType();
    if (isInc && type->isBooleanType()) {
      llvm::Value *True = CGF.EmitToMemory(Builder.getTrue(), type);
      if (isPre) {
        Builder.Insert(new llvm::StoreInst(True,
              LV.getAddress(), LV.isVolatileQualified(),
              LV.getAlignment().getQuantity(),
              llvm::SequentiallyConsistent));
        return Builder.getTrue();
      }
      // For atomic bool increment, we just store true and return it for
      // preincrement, do an atomic swap with true for postincrement
        return Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
            LV.getAddress(), True, llvm::SequentiallyConsistent);
    }
    // Special case for atomic increment / decrement on integers, emit
    // atomicrmw instructions.  We skip this if we want to be doing overflow
    // checking, and fall into the slow path with the atomic cmpxchg loop.
    if (!type->isBooleanType() && type->isIntegerType() &&
        !(type->isUnsignedIntegerType() &&
          CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) &&
        CGF.getLangOpts().getSignedOverflowBehavior() !=
            LangOptions::SOB_Trapping) {
      llvm::AtomicRMWInst::BinOp aop = isInc ? llvm::AtomicRMWInst::Add :
        llvm::AtomicRMWInst::Sub;
      llvm::Instruction::BinaryOps op = isInc ? llvm::Instruction::Add :
        llvm::Instruction::Sub;
      llvm::Value *amt = CGF.EmitToMemory(
          llvm::ConstantInt::get(ConvertType(type), 1, true), type);
      llvm::Value *old = Builder.CreateAtomicRMW(aop,
          LV.getAddress(), amt, llvm::SequentiallyConsistent);
      return isPre ? Builder.CreateBinOp(op, old, amt) : old;
    }
    value = EmitLoadOfLValue(LV, E->getExprLoc());
    input = value;
    // For every other atomic operation, we need to emit a load-op-cmpxchg loop
    llvm::BasicBlock *startBB = Builder.GetInsertBlock();
    llvm::BasicBlock *opBB = CGF.createBasicBlock("atomic_op", CGF.CurFn);
    value = CGF.EmitToMemory(value, type);
    Builder.CreateBr(opBB);
    Builder.SetInsertPoint(opBB);
    atomicPHI = Builder.CreatePHI(value->getType(), 2);
    atomicPHI->addIncoming(value, startBB);
    value = atomicPHI;
  } else {
    value = EmitLoadOfLValue(LV, E->getExprLoc());
    input = value;
  }

  // Special case of integer increment that we have to check first: bool++.
  // Due to promotion rules, we get:
  //   bool++ -> bool = bool + 1
  //          -> bool = (int)bool + 1
  //          -> bool = ((int)bool + 1 != 0)
  // An interesting aspect of this is that increment is always true.
  // Decrement does not have this property.
  if (isInc && type->isBooleanType()) {
    value = Builder.getTrue();

  // Most common case by far: integer increment.
  } else if (type->isIntegerType()) {

    llvm::Value *amt = llvm::ConstantInt::get(value->getType(), amount, true);

    // Note that signed integer inc/dec with width less than int can't
    // overflow because of promotion rules; we're just eliding a few steps here.
    bool CanOverflow = value->getType()->getIntegerBitWidth() >=
                       CGF.IntTy->getIntegerBitWidth();
    if (CanOverflow && type->isSignedIntegerOrEnumerationType()) {
      value = EmitAddConsiderOverflowBehavior(E, value, amt, isInc);
    } else if (CanOverflow && type->isUnsignedIntegerType() &&
               CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) {
      BinOpInfo BinOp;
      BinOp.LHS = value;
      BinOp.RHS = llvm::ConstantInt::get(value->getType(), 1, false);
      BinOp.Ty = E->getType();
      BinOp.Opcode = isInc ? BO_Add : BO_Sub;
      BinOp.FPContractable = false;
      BinOp.E = E;
      value = EmitOverflowCheckedBinOp(BinOp);
    } else
      value = Builder.CreateAdd(value, amt, isInc ? "inc" : "dec");

  // Next most common: pointer increment.
  } else if (const PointerType *ptr = type->getAs<PointerType>()) {
    QualType type = ptr->getPointeeType();

    // VLA types don't have constant size.
    if (const VariableArrayType *vla
          = CGF.getContext().getAsVariableArrayType(type)) {
      llvm::Value *numElts = CGF.getVLASize(vla).first;
      if (!isInc) numElts = Builder.CreateNSWNeg(numElts, "vla.negsize");
      if (CGF.getLangOpts().isSignedOverflowDefined())
        value = Builder.CreateGEP(value, numElts, "vla.inc");
      else
        value = Builder.CreateInBoundsGEP(value, numElts, "vla.inc");

    // Arithmetic on function pointers (!) is just +-1.
    } else if (type->isFunctionType()) {
      llvm::Value *amt = Builder.getInt32(amount);

      value = CGF.EmitCastToVoidPtr(value);
      if (CGF.getLangOpts().isSignedOverflowDefined())
        value = Builder.CreateGEP(value, amt, "incdec.funcptr");
      else
        value = Builder.CreateInBoundsGEP(value, amt, "incdec.funcptr");
      value = Builder.CreateBitCast(value, input->getType());

    // For everything else, we can just do a simple increment.
    } else {
      llvm::Value *amt = Builder.getInt32(amount);
      if (CGF.getLangOpts().isSignedOverflowDefined())
        value = Builder.CreateGEP(value, amt, "incdec.ptr");
      else
        value = Builder.CreateInBoundsGEP(value, amt, "incdec.ptr");
    }

  // Vector increment/decrement.
  } else if (type->isVectorType()) {
    if (type->hasIntegerRepresentation()) {
      llvm::Value *amt = llvm::ConstantInt::get(value->getType(), amount);

      value = Builder.CreateAdd(value, amt, isInc ? "inc" : "dec");
    } else {
      value = Builder.CreateFAdd(
                  value,
                  llvm::ConstantFP::get(value->getType(), amount),
                  isInc ? "inc" : "dec");
    }

  // Floating point.
  } else if (type->isRealFloatingType()) {
    // Add the inc/dec to the real part.
    llvm::Value *amt;

    if (type->isHalfType() && !CGF.getContext().getLangOpts().NativeHalfType &&
        !CGF.getContext().getLangOpts().HalfArgsAndReturns) {
      // Another special case: half FP increment should be done via float
      value = Builder.CreateCall(
          CGF.CGM.getIntrinsic(llvm::Intrinsic::convert_from_fp16,
                               CGF.CGM.FloatTy),
          input);
    }

    if (value->getType()->isFloatTy())
      amt = llvm::ConstantFP::get(VMContext,
                                  llvm::APFloat(static_cast<float>(amount)));
    else if (value->getType()->isDoubleTy())
      amt = llvm::ConstantFP::get(VMContext,
                                  llvm::APFloat(static_cast<double>(amount)));
    else {
      llvm::APFloat F(static_cast<float>(amount));
      bool ignored;
      F.convert(CGF.getTarget().getLongDoubleFormat(),
                llvm::APFloat::rmTowardZero, &ignored);
      amt = llvm::ConstantFP::get(VMContext, F);
    }
    value = Builder.CreateFAdd(value, amt, isInc ? "inc" : "dec");

    if (type->isHalfType() && !CGF.getContext().getLangOpts().NativeHalfType &&
        !CGF.getContext().getLangOpts().HalfArgsAndReturns)
      value = Builder.CreateCall(
          CGF.CGM.getIntrinsic(llvm::Intrinsic::convert_to_fp16,
                               CGF.CGM.FloatTy),
          value);

  // Objective-C pointer types.
  } else {
    const ObjCObjectPointerType *OPT = type->castAs<ObjCObjectPointerType>();
    value = CGF.EmitCastToVoidPtr(value);

    CharUnits size = CGF.getContext().getTypeSizeInChars(OPT->getObjectType());
    if (!isInc) size = -size;
    llvm::Value *sizeValue =
      llvm::ConstantInt::get(CGF.SizeTy, size.getQuantity());

    if (CGF.getLangOpts().isSignedOverflowDefined())
      value = Builder.CreateGEP(value, sizeValue, "incdec.objptr");
    else
      value = Builder.CreateInBoundsGEP(value, sizeValue, "incdec.objptr");
    value = Builder.CreateBitCast(value, input->getType());
  }

  if (atomicPHI) {
    llvm::BasicBlock *opBB = Builder.GetInsertBlock();
    llvm::BasicBlock *contBB = CGF.createBasicBlock("atomic_cont", CGF.CurFn);
    auto Pair = CGF.EmitAtomicCompareExchange(
        LV, RValue::get(atomicPHI), RValue::get(CGF.EmitToMemory(value, type)),
        E->getExprLoc());
    llvm::Value *old = Pair.first.getScalarVal();
    llvm::Value *success = Pair.second.getScalarVal();
    atomicPHI->addIncoming(old, opBB);
    Builder.CreateCondBr(success, contBB, opBB);
    Builder.SetInsertPoint(contBB);
    return isPre ? value : input;
  }

  // Store the updated result through the lvalue.
  if (LV.isBitField())
    CGF.EmitStoreThroughBitfieldLValue(RValue::get(value), LV, &value);
  else
    CGF.EmitStoreThroughLValue(RValue::get(value), LV);

  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? value : input;
}



Value *ScalarExprEmitter::VisitUnaryMinus(const UnaryOperator *E) {
  TestAndClearIgnoreResultAssign();
  // Emit unary minus with EmitSub so we handle overflow cases etc.
  BinOpInfo BinOp;
  BinOp.RHS = Visit(E->getSubExpr());

  if (BinOp.RHS->getType()->isFPOrFPVectorTy())
    BinOp.LHS = llvm::ConstantFP::getZeroValueForNegation(BinOp.RHS->getType());
  else
    BinOp.LHS = llvm::Constant::getNullValue(BinOp.RHS->getType());
  BinOp.Ty = E->getType();
  BinOp.Opcode = BO_Sub;
  BinOp.FPContractable = false;
  BinOp.E = E;
  return EmitSub(BinOp);
}

Value *ScalarExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  TestAndClearIgnoreResultAssign();
  Value *Op = Visit(E->getSubExpr());
  return Builder.CreateNot(Op, "neg");
}

Value *ScalarExprEmitter::VisitUnaryLNot(const UnaryOperator *E) {
  // Perform vector logical not on comparison with zero vector.
  if (E->getType()->isExtVectorType()) {
    Value *Oper = Visit(E->getSubExpr());
    Value *Zero = llvm::Constant::getNullValue(Oper->getType());
    Value *Result;
    if (Oper->getType()->isFPOrFPVectorTy())
      Result = Builder.CreateFCmp(llvm::CmpInst::FCMP_OEQ, Oper, Zero, "cmp");
    else
      Result = Builder.CreateICmp(llvm::CmpInst::ICMP_EQ, Oper, Zero, "cmp");
    return Builder.CreateSExt(Result, ConvertType(E->getType()), "sext");
  }

  // Compare operand to zero.
  Value *BoolVal = CGF.EvaluateExprAsBool(E->getSubExpr());

  // Invert value.
  // TODO: Could dynamically modify easy computations here.  For example, if
  // the operand is an icmp ne, turn into icmp eq.
  BoolVal = Builder.CreateNot(BoolVal, "lnot");

  // ZExt result to the expr type.
  return Builder.CreateZExt(BoolVal, ConvertType(E->getType()), "lnot.ext");
}

Value *ScalarExprEmitter::VisitOffsetOfExpr(OffsetOfExpr *E) {
  // Try folding the offsetof to a constant.
  llvm::APSInt Value;
  if (E->EvaluateAsInt(Value, CGF.getContext()))
    return Builder.getInt(Value);

  // Loop over the components of the offsetof to compute the value.
  unsigned n = E->getNumComponents();
  llvm::Type* ResultType = ConvertType(E->getType());
  llvm::Value* Result = llvm::Constant::getNullValue(ResultType);
  QualType CurrentType = E->getTypeSourceInfo()->getType();
  for (unsigned i = 0; i != n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = E->getComponent(i);
    llvm::Value *Offset = nullptr;
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array: {
      // Compute the index
      Expr *IdxExpr = E->getIndexExpr(ON.getArrayExprIndex());
      llvm::Value* Idx = CGF.EmitScalarExpr(IdxExpr);
      bool IdxSigned = IdxExpr->getType()->isSignedIntegerOrEnumerationType();
      Idx = Builder.CreateIntCast(Idx, ResultType, IdxSigned, "conv");

      // Save the element type
      CurrentType =
          CGF.getContext().getAsArrayType(CurrentType)->getElementType();

      // Compute the element size
      llvm::Value* ElemSize = llvm::ConstantInt::get(ResultType,
          CGF.getContext().getTypeSizeInChars(CurrentType).getQuantity());

      // Multiply out to compute the result
      Offset = Builder.CreateMul(Idx, ElemSize);
      break;
    }

    case OffsetOfExpr::OffsetOfNode::Field: {
      FieldDecl *MemberDecl = ON.getField();
      RecordDecl *RD = CurrentType->getAs<RecordType>()->getDecl();
      const ASTRecordLayout &RL = CGF.getContext().getASTRecordLayout(RD);

      // Compute the index of the field in its parent.
      unsigned i = 0;
      // FIXME: It would be nice if we didn't have to loop here!
      for (RecordDecl::field_iterator Field = RD->field_begin(),
                                      FieldEnd = RD->field_end();
           Field != FieldEnd; ++Field, ++i) {
        if (*Field == MemberDecl)
          break;
      }
      assert(i < RL.getFieldCount() && "offsetof field in wrong type");

      // Compute the offset to the field
      int64_t OffsetInt = RL.getFieldOffset(i) /
                          CGF.getContext().getCharWidth();
      Offset = llvm::ConstantInt::get(ResultType, OffsetInt);

      // Save the element type.
      CurrentType = MemberDecl->getType();
      break;
    }

    case OffsetOfExpr::OffsetOfNode::Identifier:
      llvm_unreachable("dependent __builtin_offsetof");

    case OffsetOfExpr::OffsetOfNode::Base: {
      if (ON.getBase()->isVirtual()) {
        CGF.ErrorUnsupported(E, "virtual base in offsetof");
        continue;
      }

      RecordDecl *RD = CurrentType->getAs<RecordType>()->getDecl();
      const ASTRecordLayout &RL = CGF.getContext().getASTRecordLayout(RD);

      // Save the element type.
      CurrentType = ON.getBase()->getType();

      // Compute the offset to the base.
      const RecordType *BaseRT = CurrentType->getAs<RecordType>();
      CXXRecordDecl *BaseRD = cast<CXXRecordDecl>(BaseRT->getDecl());
      CharUnits OffsetInt = RL.getBaseClassOffset(BaseRD);
      Offset = llvm::ConstantInt::get(ResultType, OffsetInt.getQuantity());
      break;
    }
    }
    Result = Builder.CreateAdd(Result, Offset);
  }
  return Result;
}

/// VisitUnaryExprOrTypeTraitExpr - Return the size or alignment of the type of
/// argument of the sizeof expression as an integer.
Value *
ScalarExprEmitter::VisitUnaryExprOrTypeTraitExpr(
                              const UnaryExprOrTypeTraitExpr *E) {
  QualType TypeToSize = E->getTypeOfArgument();
  if (E->getKind() == UETT_SizeOf) {
    if (const VariableArrayType *VAT =
          CGF.getContext().getAsVariableArrayType(TypeToSize)) {
      if (E->isArgumentType()) {
        // sizeof(type) - make sure to emit the VLA size.
        CGF.EmitVariablyModifiedType(TypeToSize);
      } else {
        // C99 6.5.3.4p2: If the argument is an expression of type
        // VLA, it is evaluated.
        CGF.EmitIgnoredExpr(E->getArgumentExpr());
      }

      QualType eltType;
      llvm::Value *numElts;
      std::tie(numElts, eltType) = CGF.getVLASize(VAT);

      llvm::Value *size = numElts;

      // Scale the number of non-VLA elements by the non-VLA element size.
      CharUnits eltSize = CGF.getContext().getTypeSizeInChars(eltType);
      if (!eltSize.isOne())
        size = CGF.Builder.CreateNUWMul(CGF.CGM.getSize(eltSize), numElts);

      return size;
    }
  }

  // If this isn't sizeof(vla), the result must be constant; use the constant
  // folding logic so we don't have to duplicate it here.
  return Builder.getInt(E->EvaluateKnownConstInt(CGF.getContext()));
}

Value *ScalarExprEmitter::VisitUnaryReal(const UnaryOperator *E) {
  Expr *Op = E->getSubExpr();
  if (Op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask E because Op might be an l-value that
    // this won't work for, e.g. an Obj-C property.
    if (E->isGLValue())
      return CGF.EmitLoadOfLValue(CGF.EmitLValue(E),
                                  E->getExprLoc()).getScalarVal();

    // Otherwise, calculate and project.
    return CGF.EmitComplexExpr(Op, false, true).first;
  }

  return Visit(Op);
}

Value *ScalarExprEmitter::VisitUnaryImag(const UnaryOperator *E) {
  Expr *Op = E->getSubExpr();
  if (Op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask E because Op might be an l-value that
    // this won't work for, e.g. an Obj-C property.
    if (Op->isGLValue())
      return CGF.EmitLoadOfLValue(CGF.EmitLValue(E),
                                  E->getExprLoc()).getScalarVal();

    // Otherwise, calculate and project.
    return CGF.EmitComplexExpr(Op, true, false).second;
  }

  // __imag on a scalar returns zero.  Emit the subexpr to ensure side
  // effects are evaluated, but not the actual value.
  if (Op->isGLValue())
    CGF.EmitLValue(Op);
  else
    CGF.EmitScalarExpr(Op, true);
  return llvm::Constant::getNullValue(ConvertType(E->getType()));
}

//===----------------------------------------------------------------------===//
//                           Binary Operators
//===----------------------------------------------------------------------===//

BinOpInfo ScalarExprEmitter::EmitBinOps(const BinaryOperator *E) {
  TestAndClearIgnoreResultAssign();
  BinOpInfo Result;
  Result.LHS = Visit(E->getLHS());
  Result.RHS = Visit(E->getRHS());
  Result.Ty  = E->getType();
  Result.Opcode = E->getOpcode();
  Result.FPContractable = E->isFPContractable();
  Result.E = E;
  return Result;
}

LValue ScalarExprEmitter::EmitCompoundAssignLValue(
                                              const CompoundAssignOperator *E,
                        Value *(ScalarExprEmitter::*Func)(const BinOpInfo &),
                                                   Value *&Result) {
  QualType LHSTy = E->getLHS()->getType();
  BinOpInfo OpInfo;

  if (E->getComputationResultType()->isAnyComplexType())
    return CGF.EmitScalarCompooundAssignWithComplex(E, Result);

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.
  OpInfo.RHS = Visit(E->getRHS());
  OpInfo.Ty = E->getComputationResultType();
  OpInfo.Opcode = E->getOpcode();
  OpInfo.FPContractable = false;
  OpInfo.E = E;
  // Load/convert the LHS.
  LValue LHSLV = EmitCheckedLValue(E->getLHS(), CodeGenFunction::TCK_Store);

  llvm::PHINode *atomicPHI = nullptr;
  if (const AtomicType *atomicTy = LHSTy->getAs<AtomicType>()) {
    QualType type = atomicTy->getValueType();
    if (!type->isBooleanType() && type->isIntegerType() &&
        !(type->isUnsignedIntegerType() &&
          CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) &&
        CGF.getLangOpts().getSignedOverflowBehavior() !=
            LangOptions::SOB_Trapping) {
      llvm::AtomicRMWInst::BinOp aop = llvm::AtomicRMWInst::BAD_BINOP;
      switch (OpInfo.Opcode) {
        // We don't have atomicrmw operands for *, %, /, <<, >>
        case BO_MulAssign: case BO_DivAssign:
        case BO_RemAssign:
        case BO_ShlAssign:
        case BO_ShrAssign:
          break;
        case BO_AddAssign:
          aop = llvm::AtomicRMWInst::Add;
          break;
        case BO_SubAssign:
          aop = llvm::AtomicRMWInst::Sub;
          break;
        case BO_AndAssign:
          aop = llvm::AtomicRMWInst::And;
          break;
        case BO_XorAssign:
          aop = llvm::AtomicRMWInst::Xor;
          break;
        case BO_OrAssign:
          aop = llvm::AtomicRMWInst::Or;
          break;
        default:
          llvm_unreachable("Invalid compound assignment type");
      }
      if (aop != llvm::AtomicRMWInst::BAD_BINOP) {
        llvm::Value *amt = CGF.EmitToMemory(EmitScalarConversion(OpInfo.RHS,
              E->getRHS()->getType(), LHSTy), LHSTy);
        Builder.CreateAtomicRMW(aop, LHSLV.getAddress(), amt,
            llvm::SequentiallyConsistent);
        return LHSLV;
      }
    }
    // FIXME: For floating point types, we should be saving and restoring the
    // floating point environment in the loop.
    llvm::BasicBlock *startBB = Builder.GetInsertBlock();
    llvm::BasicBlock *opBB = CGF.createBasicBlock("atomic_op", CGF.CurFn);
    OpInfo.LHS = EmitLoadOfLValue(LHSLV, E->getExprLoc());
    OpInfo.LHS = CGF.EmitToMemory(OpInfo.LHS, type);
    Builder.CreateBr(opBB);
    Builder.SetInsertPoint(opBB);
    atomicPHI = Builder.CreatePHI(OpInfo.LHS->getType(), 2);
    atomicPHI->addIncoming(OpInfo.LHS, startBB);
    OpInfo.LHS = atomicPHI;
  }
  else
    OpInfo.LHS = EmitLoadOfLValue(LHSLV, E->getExprLoc());

  OpInfo.LHS = EmitScalarConversion(OpInfo.LHS, LHSTy,
                                    E->getComputationLHSType());

  // Expand the binary operator.
  Result = (this->*Func)(OpInfo);

  // Convert the result back to the LHS type.
  Result = EmitScalarConversion(Result, E->getComputationResultType(), LHSTy);

  if (atomicPHI) {
    llvm::BasicBlock *opBB = Builder.GetInsertBlock();
    llvm::BasicBlock *contBB = CGF.createBasicBlock("atomic_cont", CGF.CurFn);
    auto Pair = CGF.EmitAtomicCompareExchange(
        LHSLV, RValue::get(atomicPHI),
        RValue::get(CGF.EmitToMemory(Result, LHSTy)), E->getExprLoc());
    llvm::Value *old = Pair.first.getScalarVal();
    llvm::Value *success = Pair.second.getScalarVal();
    atomicPHI->addIncoming(old, opBB);
    Builder.CreateCondBr(success, contBB, opBB);
    Builder.SetInsertPoint(contBB);
    return LHSLV;
  }

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (LHSLV.isBitField())
    CGF.EmitStoreThroughBitfieldLValue(RValue::get(Result), LHSLV, &Result);
  else
    CGF.EmitStoreThroughLValue(RValue::get(Result), LHSLV);

  return LHSLV;
}

Value *ScalarExprEmitter::EmitCompoundAssign(const CompoundAssignOperator *E,
                      Value *(ScalarExprEmitter::*Func)(const BinOpInfo &)) {
  bool Ignore = TestAndClearIgnoreResultAssign();
  Value *RHS;
  LValue LHS = EmitCompoundAssignLValue(E, Func, RHS);

  // If the result is clearly ignored, return now.
  if (Ignore)
    return nullptr;

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return RHS;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LHS.isVolatileQualified())
    return RHS;

  // Otherwise, reload the value.
  return EmitLoadOfLValue(LHS, E->getExprLoc());
}

void ScalarExprEmitter::EmitUndefinedBehaviorIntegerDivAndRemCheck(
    const BinOpInfo &Ops, llvm::Value *Zero, bool isDiv) {
  SmallVector<std::pair<llvm::Value *, SanitizerKind>, 2> Checks;

  if (CGF.SanOpts.has(SanitizerKind::IntegerDivideByZero)) {
    Checks.push_back(std::make_pair(Builder.CreateICmpNE(Ops.RHS, Zero),
                                    SanitizerKind::IntegerDivideByZero));
  }

  if (CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow) &&
      Ops.Ty->hasSignedIntegerRepresentation()) {
    llvm::IntegerType *Ty = cast<llvm::IntegerType>(Zero->getType());

    llvm::Value *IntMin =
      Builder.getInt(llvm::APInt::getSignedMinValue(Ty->getBitWidth()));
    llvm::Value *NegOne = llvm::ConstantInt::get(Ty, -1ULL);

    llvm::Value *LHSCmp = Builder.CreateICmpNE(Ops.LHS, IntMin);
    llvm::Value *RHSCmp = Builder.CreateICmpNE(Ops.RHS, NegOne);
    llvm::Value *NotOverflow = Builder.CreateOr(LHSCmp, RHSCmp, "or");
    Checks.push_back(
        std::make_pair(NotOverflow, SanitizerKind::SignedIntegerOverflow));
  }

  if (Checks.size() > 0)
    EmitBinOpCheck(Checks, Ops);
}

Value *ScalarExprEmitter::EmitDiv(const BinOpInfo &Ops) {
  {
    CodeGenFunction::SanitizerScope SanScope(&CGF);
    if ((CGF.SanOpts.has(SanitizerKind::IntegerDivideByZero) ||
         CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow)) &&
        Ops.Ty->isIntegerType()) {
      llvm::Value *Zero = llvm::Constant::getNullValue(ConvertType(Ops.Ty));
      EmitUndefinedBehaviorIntegerDivAndRemCheck(Ops, Zero, true);
    } else if (CGF.SanOpts.has(SanitizerKind::FloatDivideByZero) &&
               Ops.Ty->isRealFloatingType()) {
      llvm::Value *Zero = llvm::Constant::getNullValue(ConvertType(Ops.Ty));
      llvm::Value *NonZero = Builder.CreateFCmpUNE(Ops.RHS, Zero);
      EmitBinOpCheck(std::make_pair(NonZero, SanitizerKind::FloatDivideByZero),
                     Ops);
    }
  }

  if (Ops.LHS->getType()->isFPOrFPVectorTy()) {
    llvm::Value *Val = Builder.CreateFDiv(Ops.LHS, Ops.RHS, "div");
    if (CGF.getLangOpts().OpenCL) {
      // OpenCL 1.1 7.4: minimum accuracy of single precision / is 2.5ulp
      llvm::Type *ValTy = Val->getType();
      if (ValTy->isFloatTy() ||
          (isa<llvm::VectorType>(ValTy) &&
           cast<llvm::VectorType>(ValTy)->getElementType()->isFloatTy()))
        CGF.SetFPAccuracy(Val, 2.5);
    }
    return Val;
  }
  else if (Ops.Ty->hasUnsignedIntegerRepresentation())
    return Builder.CreateUDiv(Ops.LHS, Ops.RHS, "div");
  else
    return Builder.CreateSDiv(Ops.LHS, Ops.RHS, "div");
}

Value *ScalarExprEmitter::EmitRem(const BinOpInfo &Ops) {
  // Rem in C can't be a floating point type: C99 6.5.5p2.
  if (CGF.SanOpts.has(SanitizerKind::IntegerDivideByZero)) {
    CodeGenFunction::SanitizerScope SanScope(&CGF);
    llvm::Value *Zero = llvm::Constant::getNullValue(ConvertType(Ops.Ty));

    if (Ops.Ty->isIntegerType())
      EmitUndefinedBehaviorIntegerDivAndRemCheck(Ops, Zero, false);
  }

  if (Ops.Ty->hasUnsignedIntegerRepresentation())
    return Builder.CreateURem(Ops.LHS, Ops.RHS, "rem");
  else
    return Builder.CreateSRem(Ops.LHS, Ops.RHS, "rem");
}

Value *ScalarExprEmitter::EmitOverflowCheckedBinOp(const BinOpInfo &Ops) {
  unsigned IID;
  unsigned OpID = 0;

  bool isSigned = Ops.Ty->isSignedIntegerOrEnumerationType();
  switch (Ops.Opcode) {
  case BO_Add:
  case BO_AddAssign:
    OpID = 1;
    IID = isSigned ? llvm::Intrinsic::sadd_with_overflow :
                     llvm::Intrinsic::uadd_with_overflow;
    break;
  case BO_Sub:
  case BO_SubAssign:
    OpID = 2;
    IID = isSigned ? llvm::Intrinsic::ssub_with_overflow :
                     llvm::Intrinsic::usub_with_overflow;
    break;
  case BO_Mul:
  case BO_MulAssign:
    OpID = 3;
    IID = isSigned ? llvm::Intrinsic::smul_with_overflow :
                     llvm::Intrinsic::umul_with_overflow;
    break;
  default:
    llvm_unreachable("Unsupported operation for overflow detection");
  }
  OpID <<= 1;
  if (isSigned)
    OpID |= 1;

  llvm::Type *opTy = CGF.CGM.getTypes().ConvertType(Ops.Ty);

  llvm::Function *intrinsic = CGF.CGM.getIntrinsic(IID, opTy);

  Value *resultAndOverflow = Builder.CreateCall2(intrinsic, Ops.LHS, Ops.RHS);
  Value *result = Builder.CreateExtractValue(resultAndOverflow, 0);
  Value *overflow = Builder.CreateExtractValue(resultAndOverflow, 1);

  // Handle overflow with llvm.trap if no custom handler has been specified.
  const std::string *handlerName =
    &CGF.getLangOpts().OverflowHandler;
  if (handlerName->empty()) {
    // If the signed-integer-overflow sanitizer is enabled, emit a call to its
    // runtime. Otherwise, this is a -ftrapv check, so just emit a trap.
    if (!isSigned || CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow)) {
      CodeGenFunction::SanitizerScope SanScope(&CGF);
      llvm::Value *NotOverflow = Builder.CreateNot(overflow);
      SanitizerKind Kind = isSigned ? SanitizerKind::SignedIntegerOverflow
                              : SanitizerKind::UnsignedIntegerOverflow;
      EmitBinOpCheck(std::make_pair(NotOverflow, Kind), Ops);
    } else
      CGF.EmitTrapCheck(Builder.CreateNot(overflow));
    return result;
  }

  // Branch in case of overflow.
  llvm::BasicBlock *initialBB = Builder.GetInsertBlock();
  llvm::Function::iterator insertPt = initialBB;
  llvm::BasicBlock *continueBB = CGF.createBasicBlock("nooverflow", CGF.CurFn,
                                                      std::next(insertPt));
  llvm::BasicBlock *overflowBB = CGF.createBasicBlock("overflow", CGF.CurFn);

  Builder.CreateCondBr(overflow, overflowBB, continueBB);

  // If an overflow handler is set, then we want to call it and then use its
  // result, if it returns.
  Builder.SetInsertPoint(overflowBB);

  // Get the overflow handler.
  llvm::Type *Int8Ty = CGF.Int8Ty;
  llvm::Type *argTypes[] = { CGF.Int64Ty, CGF.Int64Ty, Int8Ty, Int8Ty };
  llvm::FunctionType *handlerTy =
      llvm::FunctionType::get(CGF.Int64Ty, argTypes, true);
  llvm::Value *handler = CGF.CGM.CreateRuntimeFunction(handlerTy, *handlerName);

  // Sign extend the args to 64-bit, so that we can use the same handler for
  // all types of overflow.
  llvm::Value *lhs = Builder.CreateSExt(Ops.LHS, CGF.Int64Ty);
  llvm::Value *rhs = Builder.CreateSExt(Ops.RHS, CGF.Int64Ty);

  // Call the handler with the two arguments, the operation, and the size of
  // the result.
  llvm::Value *handlerArgs[] = {
    lhs,
    rhs,
    Builder.getInt8(OpID),
    Builder.getInt8(cast<llvm::IntegerType>(opTy)->getBitWidth())
  };
  llvm::Value *handlerResult =
    CGF.EmitNounwindRuntimeCall(handler, handlerArgs);

  // Truncate the result back to the desired size.
  handlerResult = Builder.CreateTrunc(handlerResult, opTy);
  Builder.CreateBr(continueBB);

  Builder.SetInsertPoint(continueBB);
  llvm::PHINode *phi = Builder.CreatePHI(opTy, 2);
  phi->addIncoming(result, initialBB);
  phi->addIncoming(handlerResult, overflowBB);

  return phi;
}

/// Emit pointer + index arithmetic.
static Value *emitPointerArithmetic(CodeGenFunction &CGF,
                                    const BinOpInfo &op,
                                    bool isSubtraction) {
  // Must have binary (not unary) expr here.  Unary pointer
  // increment/decrement doesn't use this path.
  const BinaryOperator *expr = cast<BinaryOperator>(op.E);

  Value *pointer = op.LHS;
  Expr *pointerOperand = expr->getLHS();
  Value *index = op.RHS;
  Expr *indexOperand = expr->getRHS();

  // In a subtraction, the LHS is always the pointer.
  if (!isSubtraction && !pointer->getType()->isPointerTy()) {
    std::swap(pointer, index);
    std::swap(pointerOperand, indexOperand);
  }

  unsigned width = cast<llvm::IntegerType>(index->getType())->getBitWidth();
  if (width != CGF.PointerWidthInBits) {
    // Zero-extend or sign-extend the pointer value according to
    // whether the index is signed or not.
    bool isSigned = indexOperand->getType()->isSignedIntegerOrEnumerationType();
    index = CGF.Builder.CreateIntCast(index, CGF.PtrDiffTy, isSigned,
                                      "idx.ext");
  }

  // If this is subtraction, negate the index.
  if (isSubtraction)
    index = CGF.Builder.CreateNeg(index, "idx.neg");

  if (CGF.SanOpts.has(SanitizerKind::ArrayBounds))
    CGF.EmitBoundsCheck(op.E, pointerOperand, index, indexOperand->getType(),
                        /*Accessed*/ false);

  const PointerType *pointerType
    = pointerOperand->getType()->getAs<PointerType>();
  if (!pointerType) {
    QualType objectType = pointerOperand->getType()
                                        ->castAs<ObjCObjectPointerType>()
                                        ->getPointeeType();
    llvm::Value *objectSize
      = CGF.CGM.getSize(CGF.getContext().getTypeSizeInChars(objectType));

    index = CGF.Builder.CreateMul(index, objectSize);

    Value *result = CGF.Builder.CreateBitCast(pointer, CGF.VoidPtrTy);
    result = CGF.Builder.CreateGEP(result, index, "add.ptr");
    return CGF.Builder.CreateBitCast(result, pointer->getType());
  }

  QualType elementType = pointerType->getPointeeType();
  if (const VariableArrayType *vla
        = CGF.getContext().getAsVariableArrayType(elementType)) {
    // The element count here is the total number of non-VLA elements.
    llvm::Value *numElements = CGF.getVLASize(vla).first;

    // Effectively, the multiply by the VLA size is part of the GEP.
    // GEP indexes are signed, and scaling an index isn't permitted to
    // signed-overflow, so we use the same semantics for our explicit
    // multiply.  We suppress this if overflow is not undefined behavior.
    if (CGF.getLangOpts().isSignedOverflowDefined()) {
      index = CGF.Builder.CreateMul(index, numElements, "vla.index");
      pointer = CGF.Builder.CreateGEP(pointer, index, "add.ptr");
    } else {
      index = CGF.Builder.CreateNSWMul(index, numElements, "vla.index");
      pointer = CGF.Builder.CreateInBoundsGEP(pointer, index, "add.ptr");
    }
    return pointer;
  }

  // Explicitly handle GNU void* and function pointer arithmetic extensions. The
  // GNU void* casts amount to no-ops since our void* type is i8*, but this is
  // future proof.
  if (elementType->isVoidType() || elementType->isFunctionType()) {
    Value *result = CGF.Builder.CreateBitCast(pointer, CGF.VoidPtrTy);
    result = CGF.Builder.CreateGEP(result, index, "add.ptr");
    return CGF.Builder.CreateBitCast(result, pointer->getType());
  }

  if (CGF.getLangOpts().isSignedOverflowDefined())
    return CGF.Builder.CreateGEP(pointer, index, "add.ptr");

  return CGF.Builder.CreateInBoundsGEP(pointer, index, "add.ptr");
}

// Construct an fmuladd intrinsic to represent a fused mul-add of MulOp and
// Addend. Use negMul and negAdd to negate the first operand of the Mul or
// the add operand respectively. This allows fmuladd to represent a*b-c, or
// c-a*b. Patterns in LLVM should catch the negated forms and translate them to
// efficient operations.
static Value* buildFMulAdd(llvm::BinaryOperator *MulOp, Value *Addend,
                           const CodeGenFunction &CGF, CGBuilderTy &Builder,
                           bool negMul, bool negAdd) {
  assert(!(negMul && negAdd) && "Only one of negMul and negAdd should be set.");

  Value *MulOp0 = MulOp->getOperand(0);
  Value *MulOp1 = MulOp->getOperand(1);
  if (negMul) {
    MulOp0 =
      Builder.CreateFSub(
        llvm::ConstantFP::getZeroValueForNegation(MulOp0->getType()), MulOp0,
        "neg");
  } else if (negAdd) {
    Addend =
      Builder.CreateFSub(
        llvm::ConstantFP::getZeroValueForNegation(Addend->getType()), Addend,
        "neg");
  }

  Value *FMulAdd =
    Builder.CreateCall3(
      CGF.CGM.getIntrinsic(llvm::Intrinsic::fmuladd, Addend->getType()),
                           MulOp0, MulOp1, Addend);
   MulOp->eraseFromParent();

   return FMulAdd;
}

// Check whether it would be legal to emit an fmuladd intrinsic call to
// represent op and if so, build the fmuladd.
//
// Checks that (a) the operation is fusable, and (b) -ffp-contract=on.
// Does NOT check the type of the operation - it's assumed that this function
// will be called from contexts where it's known that the type is contractable.
static Value* tryEmitFMulAdd(const BinOpInfo &op,
                         const CodeGenFunction &CGF, CGBuilderTy &Builder,
                         bool isSub=false) {

  assert((op.Opcode == BO_Add || op.Opcode == BO_AddAssign ||
          op.Opcode == BO_Sub || op.Opcode == BO_SubAssign) &&
         "Only fadd/fsub can be the root of an fmuladd.");

  // Check whether this op is marked as fusable.
  if (!op.FPContractable)
    return nullptr;

  // Check whether -ffp-contract=on. (If -ffp-contract=off/fast, fusing is
  // either disabled, or handled entirely by the LLVM backend).
  if (CGF.CGM.getCodeGenOpts().getFPContractMode() != CodeGenOptions::FPC_On)
    return nullptr;

  // We have a potentially fusable op. Look for a mul on one of the operands.
  if (llvm::BinaryOperator* LHSBinOp = dyn_cast<llvm::BinaryOperator>(op.LHS)) {
    if (LHSBinOp->getOpcode() == llvm::Instruction::FMul) {
      assert(LHSBinOp->getNumUses() == 0 &&
             "Operations with multiple uses shouldn't be contracted.");
      return buildFMulAdd(LHSBinOp, op.RHS, CGF, Builder, false, isSub);
    }
  } else if (llvm::BinaryOperator* RHSBinOp =
               dyn_cast<llvm::BinaryOperator>(op.RHS)) {
    if (RHSBinOp->getOpcode() == llvm::Instruction::FMul) {
      assert(RHSBinOp->getNumUses() == 0 &&
             "Operations with multiple uses shouldn't be contracted.");
      return buildFMulAdd(RHSBinOp, op.LHS, CGF, Builder, isSub, false);
    }
  }

  return nullptr;
}

Value *ScalarExprEmitter::EmitAdd(const BinOpInfo &op) {
  if (op.LHS->getType()->isPointerTy() ||
      op.RHS->getType()->isPointerTy())
    return emitPointerArithmetic(CGF, op, /*subtraction*/ false);

  if (op.Ty->isSignedIntegerOrEnumerationType()) {
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      return Builder.CreateAdd(op.LHS, op.RHS, "add");
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.CreateNSWAdd(op.LHS, op.RHS, "add");
      // Fall through.
    case LangOptions::SOB_Trapping:
      return EmitOverflowCheckedBinOp(op);
    }
  }

  if (op.Ty->isUnsignedIntegerType() &&
      CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow))
    return EmitOverflowCheckedBinOp(op);

  if (op.LHS->getType()->isFPOrFPVectorTy()) {
    // Try to form an fmuladd.
    if (Value *FMulAdd = tryEmitFMulAdd(op, CGF, Builder))
      return FMulAdd;

    return Builder.CreateFAdd(op.LHS, op.RHS, "add");
  }

  return Builder.CreateAdd(op.LHS, op.RHS, "add");
}

Value *ScalarExprEmitter::EmitSub(const BinOpInfo &op) {
  // The LHS is always a pointer if either side is.
  if (!op.LHS->getType()->isPointerTy()) {
    if (op.Ty->isSignedIntegerOrEnumerationType()) {
      switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
      case LangOptions::SOB_Defined:
        return Builder.CreateSub(op.LHS, op.RHS, "sub");
      case LangOptions::SOB_Undefined:
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.CreateNSWSub(op.LHS, op.RHS, "sub");
        // Fall through.
      case LangOptions::SOB_Trapping:
        return EmitOverflowCheckedBinOp(op);
      }
    }

    if (op.Ty->isUnsignedIntegerType() &&
        CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow))
      return EmitOverflowCheckedBinOp(op);

    if (op.LHS->getType()->isFPOrFPVectorTy()) {
      // Try to form an fmuladd.
      if (Value *FMulAdd = tryEmitFMulAdd(op, CGF, Builder, true))
        return FMulAdd;
      return Builder.CreateFSub(op.LHS, op.RHS, "sub");
    }

    return Builder.CreateSub(op.LHS, op.RHS, "sub");
  }

  // If the RHS is not a pointer, then we have normal pointer
  // arithmetic.
  if (!op.RHS->getType()->isPointerTy())
    return emitPointerArithmetic(CGF, op, /*subtraction*/ true);

  // Otherwise, this is a pointer subtraction.

  // Do the raw subtraction part.
  llvm::Value *LHS
    = Builder.CreatePtrToInt(op.LHS, CGF.PtrDiffTy, "sub.ptr.lhs.cast");
  llvm::Value *RHS
    = Builder.CreatePtrToInt(op.RHS, CGF.PtrDiffTy, "sub.ptr.rhs.cast");
  Value *diffInChars = Builder.CreateSub(LHS, RHS, "sub.ptr.sub");

  // Okay, figure out the element size.
  const BinaryOperator *expr = cast<BinaryOperator>(op.E);
  QualType elementType = expr->getLHS()->getType()->getPointeeType();

  llvm::Value *divisor = nullptr;

  // For a variable-length array, this is going to be non-constant.
  if (const VariableArrayType *vla
        = CGF.getContext().getAsVariableArrayType(elementType)) {
    llvm::Value *numElements;
    std::tie(numElements, elementType) = CGF.getVLASize(vla);

    divisor = numElements;

    // Scale the number of non-VLA elements by the non-VLA element size.
    CharUnits eltSize = CGF.getContext().getTypeSizeInChars(elementType);
    if (!eltSize.isOne())
      divisor = CGF.Builder.CreateNUWMul(CGF.CGM.getSize(eltSize), divisor);

  // For everything elese, we can just compute it, safe in the
  // assumption that Sema won't let anything through that we can't
  // safely compute the size of.
  } else {
    CharUnits elementSize;
    // Handle GCC extension for pointer arithmetic on void* and
    // function pointer types.
    if (elementType->isVoidType() || elementType->isFunctionType())
      elementSize = CharUnits::One();
    else
      elementSize = CGF.getContext().getTypeSizeInChars(elementType);

    // Don't even emit the divide for element size of 1.
    if (elementSize.isOne())
      return diffInChars;

    divisor = CGF.CGM.getSize(elementSize);
  }

  // Otherwise, do a full sdiv. This uses the "exact" form of sdiv, since
  // pointer difference in C is only defined in the case where both operands
  // are pointing to elements of an array.
  return Builder.CreateExactSDiv(diffInChars, divisor, "sub.ptr.div");
}

Value *ScalarExprEmitter::GetWidthMinusOneValue(Value* LHS,Value* RHS) {
  llvm::IntegerType *Ty;
  if (llvm::VectorType *VT = dyn_cast<llvm::VectorType>(LHS->getType()))
    Ty = cast<llvm::IntegerType>(VT->getElementType());
  else
    Ty = cast<llvm::IntegerType>(LHS->getType());
  return llvm::ConstantInt::get(RHS->getType(), Ty->getBitWidth() - 1);
}

Value *ScalarExprEmitter::EmitShl(const BinOpInfo &Ops) {
  // LLVM requires the LHS and RHS to be the same type: promote or truncate the
  // RHS to the same size as the LHS.
  Value *RHS = Ops.RHS;
  if (Ops.LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, Ops.LHS->getType(), false, "sh_prom");

  if (CGF.SanOpts.has(SanitizerKind::Shift) && !CGF.getLangOpts().OpenCL &&
      isa<llvm::IntegerType>(Ops.LHS->getType())) {
    CodeGenFunction::SanitizerScope SanScope(&CGF);
    llvm::Value *WidthMinusOne = GetWidthMinusOneValue(Ops.LHS, RHS);
    llvm::Value *Valid = Builder.CreateICmpULE(RHS, WidthMinusOne);

    if (Ops.Ty->hasSignedIntegerRepresentation()) {
      llvm::BasicBlock *Orig = Builder.GetInsertBlock();
      llvm::BasicBlock *Cont = CGF.createBasicBlock("cont");
      llvm::BasicBlock *CheckBitsShifted = CGF.createBasicBlock("check");
      Builder.CreateCondBr(Valid, CheckBitsShifted, Cont);

      // Check whether we are shifting any non-zero bits off the top of the
      // integer.
      CGF.EmitBlock(CheckBitsShifted);
      llvm::Value *BitsShiftedOff =
        Builder.CreateLShr(Ops.LHS,
                           Builder.CreateSub(WidthMinusOne, RHS, "shl.zeros",
                                             /*NUW*/true, /*NSW*/true),
                           "shl.check");
      if (CGF.getLangOpts().CPlusPlus) {
        // In C99, we are not permitted to shift a 1 bit into the sign bit.
        // Under C++11's rules, shifting a 1 bit into the sign bit is
        // OK, but shifting a 1 bit out of it is not. (C89 and C++03 don't
        // define signed left shifts, so we use the C99 and C++11 rules there).
        llvm::Value *One = llvm::ConstantInt::get(BitsShiftedOff->getType(), 1);
        BitsShiftedOff = Builder.CreateLShr(BitsShiftedOff, One);
      }
      llvm::Value *Zero = llvm::ConstantInt::get(BitsShiftedOff->getType(), 0);
      llvm::Value *SecondCheck = Builder.CreateICmpEQ(BitsShiftedOff, Zero);
      CGF.EmitBlock(Cont);
      llvm::PHINode *P = Builder.CreatePHI(Valid->getType(), 2);
      P->addIncoming(Valid, Orig);
      P->addIncoming(SecondCheck, CheckBitsShifted);
      Valid = P;
    }

    EmitBinOpCheck(std::make_pair(Valid, SanitizerKind::Shift), Ops);
  }
  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    RHS = Builder.CreateAnd(RHS, GetWidthMinusOneValue(Ops.LHS, RHS), "shl.mask");

  return Builder.CreateShl(Ops.LHS, RHS, "shl");
}

Value *ScalarExprEmitter::EmitShr(const BinOpInfo &Ops) {
  // LLVM requires the LHS and RHS to be the same type: promote or truncate the
  // RHS to the same size as the LHS.
  Value *RHS = Ops.RHS;
  if (Ops.LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, Ops.LHS->getType(), false, "sh_prom");

  if (CGF.SanOpts.has(SanitizerKind::Shift) && !CGF.getLangOpts().OpenCL &&
      isa<llvm::IntegerType>(Ops.LHS->getType())) {
    CodeGenFunction::SanitizerScope SanScope(&CGF);
    llvm::Value *Valid =
        Builder.CreateICmpULE(RHS, GetWidthMinusOneValue(Ops.LHS, RHS));
    EmitBinOpCheck(std::make_pair(Valid, SanitizerKind::Shift), Ops);
  }

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    RHS = Builder.CreateAnd(RHS, GetWidthMinusOneValue(Ops.LHS, RHS), "shr.mask");

  if (Ops.Ty->hasUnsignedIntegerRepresentation())
    return Builder.CreateLShr(Ops.LHS, RHS, "shr");
  return Builder.CreateAShr(Ops.LHS, RHS, "shr");
}

enum IntrinsicType { VCMPEQ, VCMPGT };
// return corresponding comparison intrinsic for given vector type
static llvm::Intrinsic::ID GetIntrinsic(IntrinsicType IT,
                                        BuiltinType::Kind ElemKind) {
  switch (ElemKind) {
  default: llvm_unreachable("unexpected element type");
  case BuiltinType::Char_U:
  case BuiltinType::UChar:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpequb_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtub_p;
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpequb_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtsb_p;
  case BuiltinType::UShort:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpequh_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtuh_p;
  case BuiltinType::Short:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpequh_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtsh_p;
  case BuiltinType::UInt:
  case BuiltinType::ULong:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpequw_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtuw_p;
  case BuiltinType::Int:
  case BuiltinType::Long:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpequw_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtsw_p;
  case BuiltinType::Float:
    return (IT == VCMPEQ) ? llvm::Intrinsic::ppc_altivec_vcmpeqfp_p :
                            llvm::Intrinsic::ppc_altivec_vcmpgtfp_p;
  }
}

Value *ScalarExprEmitter::EmitCompare(const BinaryOperator *E,unsigned UICmpOpc,
                                      unsigned SICmpOpc, unsigned FCmpOpc) {
  TestAndClearIgnoreResultAssign();
  Value *Result;
  QualType LHSTy = E->getLHS()->getType();
  QualType RHSTy = E->getRHS()->getType();
  if (const MemberPointerType *MPT = LHSTy->getAs<MemberPointerType>()) {
    assert(E->getOpcode() == BO_EQ ||
           E->getOpcode() == BO_NE);
    Value *LHS = CGF.EmitScalarExpr(E->getLHS());
    Value *RHS = CGF.EmitScalarExpr(E->getRHS());
    Result = CGF.CGM.getCXXABI().EmitMemberPointerComparison(
                   CGF, LHS, RHS, MPT, E->getOpcode() == BO_NE);
  } else if (!LHSTy->isAnyComplexType() && !RHSTy->isAnyComplexType()) {
    Value *LHS = Visit(E->getLHS());
    Value *RHS = Visit(E->getRHS());

    // If AltiVec, the comparison results in a numeric type, so we use
    // intrinsics comparing vectors and giving 0 or 1 as a result
    if (LHSTy->isVectorType() && !E->getType()->isVectorType()) {
      // constants for mapping CR6 register bits to predicate result
      enum { CR6_EQ=0, CR6_EQ_REV, CR6_LT, CR6_LT_REV } CR6;

      llvm::Intrinsic::ID ID = llvm::Intrinsic::not_intrinsic;

      // in several cases vector arguments order will be reversed
      Value *FirstVecArg = LHS,
            *SecondVecArg = RHS;

      QualType ElTy = LHSTy->getAs<VectorType>()->getElementType();
      const BuiltinType *BTy = ElTy->getAs<BuiltinType>();
      BuiltinType::Kind ElementKind = BTy->getKind();

      switch(E->getOpcode()) {
      default: llvm_unreachable("is not a comparison operation");
      case BO_EQ:
        CR6 = CR6_LT;
        ID = GetIntrinsic(VCMPEQ, ElementKind);
        break;
      case BO_NE:
        CR6 = CR6_EQ;
        ID = GetIntrinsic(VCMPEQ, ElementKind);
        break;
      case BO_LT:
        CR6 = CR6_LT;
        ID = GetIntrinsic(VCMPGT, ElementKind);
        std::swap(FirstVecArg, SecondVecArg);
        break;
      case BO_GT:
        CR6 = CR6_LT;
        ID = GetIntrinsic(VCMPGT, ElementKind);
        break;
      case BO_LE:
        if (ElementKind == BuiltinType::Float) {
          CR6 = CR6_LT;
          ID = llvm::Intrinsic::ppc_altivec_vcmpgefp_p;
          std::swap(FirstVecArg, SecondVecArg);
        }
        else {
          CR6 = CR6_EQ;
          ID = GetIntrinsic(VCMPGT, ElementKind);
        }
        break;
      case BO_GE:
        if (ElementKind == BuiltinType::Float) {
          CR6 = CR6_LT;
          ID = llvm::Intrinsic::ppc_altivec_vcmpgefp_p;
        }
        else {
          CR6 = CR6_EQ;
          ID = GetIntrinsic(VCMPGT, ElementKind);
          std::swap(FirstVecArg, SecondVecArg);
        }
        break;
      }

      Value *CR6Param = Builder.getInt32(CR6);
      llvm::Function *F = CGF.CGM.getIntrinsic(ID);
      Result = Builder.CreateCall3(F, CR6Param, FirstVecArg, SecondVecArg, "");
      return EmitScalarConversion(Result, CGF.getContext().BoolTy, E->getType());
    }

    if (LHS->getType()->isFPOrFPVectorTy()) {
      Result = Builder.CreateFCmp((llvm::CmpInst::Predicate)FCmpOpc,
                                  LHS, RHS, "cmp");
    } else if (LHSTy->hasSignedIntegerRepresentation()) {
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)SICmpOpc,
                                  LHS, RHS, "cmp");
    } else {
      // Unsigned integers and pointers.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                  LHS, RHS, "cmp");
    }

    // If this is a vector comparison, sign extend the result to the appropriate
    // vector integer type and return it (don't convert to bool).
    if (LHSTy->isVectorType())
      return Builder.CreateSExt(Result, ConvertType(E->getType()), "sext");

  } else {
    // Complex Comparison: can only be an equality comparison.
    CodeGenFunction::ComplexPairTy LHS, RHS;
    QualType CETy;
    if (auto *CTy = LHSTy->getAs<ComplexType>()) {
      LHS = CGF.EmitComplexExpr(E->getLHS());
      CETy = CTy->getElementType();
    } else {
      LHS.first = Visit(E->getLHS());
      LHS.second = llvm::Constant::getNullValue(LHS.first->getType());
      CETy = LHSTy;
    }
    if (auto *CTy = RHSTy->getAs<ComplexType>()) {
      RHS = CGF.EmitComplexExpr(E->getRHS());
      assert(CGF.getContext().hasSameUnqualifiedType(CETy,
                                                     CTy->getElementType()) &&
             "The element types must always match.");
      (void)CTy;
    } else {
      RHS.first = Visit(E->getRHS());
      RHS.second = llvm::Constant::getNullValue(RHS.first->getType());
      assert(CGF.getContext().hasSameUnqualifiedType(CETy, RHSTy) &&
             "The element types must always match.");
    }

    Value *ResultR, *ResultI;
    if (CETy->isRealFloatingType()) {
      ResultR = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                   LHS.first, RHS.first, "cmp.r");
      ResultI = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                   LHS.second, RHS.second, "cmp.i");
    } else {
      // Complex comparisons can only be equality comparisons.  As such, signed
      // and unsigned opcodes are the same.
      ResultR = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                   LHS.first, RHS.first, "cmp.r");
      ResultI = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                   LHS.second, RHS.second, "cmp.i");
    }

    if (E->getOpcode() == BO_EQ) {
      Result = Builder.CreateAnd(ResultR, ResultI, "and.ri");
    } else {
      assert(E->getOpcode() == BO_NE &&
             "Complex comparison other than == or != ?");
      Result = Builder.CreateOr(ResultR, ResultI, "or.ri");
    }
  }

  return EmitScalarConversion(Result, CGF.getContext().BoolTy, E->getType());
}

Value *ScalarExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();

  Value *RHS;
  LValue LHS;

  switch (E->getLHS()->getType().getObjCLifetime()) {
  case Qualifiers::OCL_Strong:
    std::tie(LHS, RHS) = CGF.EmitARCStoreStrong(E, Ignore);
    break;

  case Qualifiers::OCL_Autoreleasing:
    std::tie(LHS, RHS) = CGF.EmitARCStoreAutoreleasing(E);
    break;

  case Qualifiers::OCL_Weak:
    RHS = Visit(E->getRHS());
    LHS = EmitCheckedLValue(E->getLHS(), CodeGenFunction::TCK_Store);
    RHS = CGF.EmitARCStoreWeak(LHS.getAddress(), RHS, Ignore);
    break;

  // No reason to do any of these differently.
  case Qualifiers::OCL_None:
  case Qualifiers::OCL_ExplicitNone:
    // __block variables need to have the rhs evaluated first, plus
    // this should improve codegen just a little.
    RHS = Visit(E->getRHS());
    LHS = EmitCheckedLValue(E->getLHS(), CodeGenFunction::TCK_Store);

    // Store the value into the LHS.  Bit-fields are handled specially
    // because the result is altered by the store, i.e., [C99 6.5.16p1]
    // 'An assignment expression has the value of the left operand after
    // the assignment...'.
    if (LHS.isBitField())
      CGF.EmitStoreThroughBitfieldLValue(RValue::get(RHS), LHS, &RHS);
    else
      CGF.EmitStoreThroughLValue(RValue::get(RHS), LHS);
  }

  // If the result is clearly ignored, return now.
  if (Ignore)
    return nullptr;

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return RHS;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LHS.isVolatileQualified())
    return RHS;

  // Otherwise, reload the value.
  return EmitLoadOfLValue(LHS, E->getExprLoc());
}

Value *ScalarExprEmitter::VisitBinLAnd(const BinaryOperator *E) {
  RegionCounter Cnt = CGF.getPGORegionCounter(E);

  // Perform vector logical and on comparisons with zero vectors.
  if (E->getType()->isVectorType()) {
    Cnt.beginRegion(Builder);

    Value *LHS = Visit(E->getLHS());
    Value *RHS = Visit(E->getRHS());
    Value *Zero = llvm::ConstantAggregateZero::get(LHS->getType());
    if (LHS->getType()->isFPOrFPVectorTy()) {
      LHS = Builder.CreateFCmp(llvm::CmpInst::FCMP_UNE, LHS, Zero, "cmp");
      RHS = Builder.CreateFCmp(llvm::CmpInst::FCMP_UNE, RHS, Zero, "cmp");
    } else {
      LHS = Builder.CreateICmp(llvm::CmpInst::ICMP_NE, LHS, Zero, "cmp");
      RHS = Builder.CreateICmp(llvm::CmpInst::ICMP_NE, RHS, Zero, "cmp");
    }
    Value *And = Builder.CreateAnd(LHS, RHS);
    return Builder.CreateSExt(And, ConvertType(E->getType()), "sext");
  }

  llvm::Type *ResTy = ConvertType(E->getType());

  // If we have 0 && RHS, see if we can elide RHS, if so, just return 0.
  // If we have 1 && X, just emit X without inserting the control flow.
  bool LHSCondVal;
  if (CGF.ConstantFoldsToSimpleInteger(E->getLHS(), LHSCondVal)) {
    if (LHSCondVal) { // If we have 1 && X, just emit X.
      Cnt.beginRegion(Builder);

      Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
      // ZExt result to int or bool.
      return Builder.CreateZExtOrBitCast(RHSCond, ResTy, "land.ext");
    }

    // 0 && RHS: If it is safe, just elide the RHS, and return 0/false.
    if (!CGF.ContainsLabel(E->getRHS()))
      return llvm::Constant::getNullValue(ResTy);
  }

  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("land.end");
  llvm::BasicBlock *RHSBlock  = CGF.createBasicBlock("land.rhs");

  CodeGenFunction::ConditionalEvaluation eval(CGF);

  // Branch on the LHS first.  If it is false, go to the failure (cont) block.
  CGF.EmitBranchOnBoolExpr(E->getLHS(), RHSBlock, ContBlock, Cnt.getCount());

  // Any edges into the ContBlock are now from an (indeterminate number of)
  // edges from this first condition.  All of these values will be false.  Start
  // setting up the PHI node in the Cont Block for this.
  llvm::PHINode *PN = llvm::PHINode::Create(llvm::Type::getInt1Ty(VMContext), 2,
                                            "", ContBlock);
  for (llvm::pred_iterator PI = pred_begin(ContBlock), PE = pred_end(ContBlock);
       PI != PE; ++PI)
    PN->addIncoming(llvm::ConstantInt::getFalse(VMContext), *PI);

  eval.begin(CGF);
  CGF.EmitBlock(RHSBlock);
  Cnt.beginRegion(Builder);
  Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
  eval.end(CGF);

  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();

  // Emit an unconditional branch from this block to ContBlock.
  {
    // There is no need to emit line number for unconditional branch.
    ApplyDebugLocation DL(CGF);
    CGF.EmitBlock(ContBlock);
  }
  // Insert an entry into the phi node for the edge with the value of RHSCond.
  PN->addIncoming(RHSCond, RHSBlock);

  // ZExt result to int.
  return Builder.CreateZExtOrBitCast(PN, ResTy, "land.ext");
}

Value *ScalarExprEmitter::VisitBinLOr(const BinaryOperator *E) {
  RegionCounter Cnt = CGF.getPGORegionCounter(E);

  // Perform vector logical or on comparisons with zero vectors.
  if (E->getType()->isVectorType()) {
    Cnt.beginRegion(Builder);

    Value *LHS = Visit(E->getLHS());
    Value *RHS = Visit(E->getRHS());
    Value *Zero = llvm::ConstantAggregateZero::get(LHS->getType());
    if (LHS->getType()->isFPOrFPVectorTy()) {
      LHS = Builder.CreateFCmp(llvm::CmpInst::FCMP_UNE, LHS, Zero, "cmp");
      RHS = Builder.CreateFCmp(llvm::CmpInst::FCMP_UNE, RHS, Zero, "cmp");
    } else {
      LHS = Builder.CreateICmp(llvm::CmpInst::ICMP_NE, LHS, Zero, "cmp");
      RHS = Builder.CreateICmp(llvm::CmpInst::ICMP_NE, RHS, Zero, "cmp");
    }
    Value *Or = Builder.CreateOr(LHS, RHS);
    return Builder.CreateSExt(Or, ConvertType(E->getType()), "sext");
  }

  llvm::Type *ResTy = ConvertType(E->getType());

  // If we have 1 || RHS, see if we can elide RHS, if so, just return 1.
  // If we have 0 || X, just emit X without inserting the control flow.
  bool LHSCondVal;
  if (CGF.ConstantFoldsToSimpleInteger(E->getLHS(), LHSCondVal)) {
    if (!LHSCondVal) { // If we have 0 || X, just emit X.
      Cnt.beginRegion(Builder);

      Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
      // ZExt result to int or bool.
      return Builder.CreateZExtOrBitCast(RHSCond, ResTy, "lor.ext");
    }

    // 1 || RHS: If it is safe, just elide the RHS, and return 1/true.
    if (!CGF.ContainsLabel(E->getRHS()))
      return llvm::ConstantInt::get(ResTy, 1);
  }

  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("lor.end");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("lor.rhs");

  CodeGenFunction::ConditionalEvaluation eval(CGF);

  // Branch on the LHS first.  If it is true, go to the success (cont) block.
  CGF.EmitBranchOnBoolExpr(E->getLHS(), ContBlock, RHSBlock,
                           Cnt.getParentCount() - Cnt.getCount());

  // Any edges into the ContBlock are now from an (indeterminate number of)
  // edges from this first condition.  All of these values will be true.  Start
  // setting up the PHI node in the Cont Block for this.
  llvm::PHINode *PN = llvm::PHINode::Create(llvm::Type::getInt1Ty(VMContext), 2,
                                            "", ContBlock);
  for (llvm::pred_iterator PI = pred_begin(ContBlock), PE = pred_end(ContBlock);
       PI != PE; ++PI)
    PN->addIncoming(llvm::ConstantInt::getTrue(VMContext), *PI);

  eval.begin(CGF);

  // Emit the RHS condition as a bool value.
  CGF.EmitBlock(RHSBlock);
  Cnt.beginRegion(Builder);
  Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());

  eval.end(CGF);

  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();

  // Emit an unconditional branch from this block to ContBlock.  Insert an entry
  // into the phi node for the edge with the value of RHSCond.
  CGF.EmitBlock(ContBlock);
  PN->addIncoming(RHSCond, RHSBlock);

  // ZExt result to int.
  return Builder.CreateZExtOrBitCast(PN, ResTy, "lor.ext");
}

Value *ScalarExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitIgnoredExpr(E->getLHS());
  CGF.EnsureInsertPoint();
  return Visit(E->getRHS());
}

//===----------------------------------------------------------------------===//
//                             Other Operators
//===----------------------------------------------------------------------===//

/// isCheapEnoughToEvaluateUnconditionally - Return true if the specified
/// expression is cheap enough and side-effect-free enough to evaluate
/// unconditionally instead of conditionally.  This is used to convert control
/// flow into selects in some cases.
static bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                                   CodeGenFunction &CGF) {
  // Anything that is an integer or floating point constant is fine.
  return E->IgnoreParens()->isEvaluatable(CGF.getContext());

  // Even non-volatile automatic variables can't be evaluated unconditionally.
  // Referencing a thread_local may cause non-trivial initialization work to
  // occur. If we're inside a lambda and one of the variables is from the scope
  // outside the lambda, that function may have returned already. Reading its
  // locals is a bad idea. Also, these reads may introduce races there didn't
  // exist in the source-level program.
}


Value *ScalarExprEmitter::
VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
  TestAndClearIgnoreResultAssign();

  // Bind the common expression if necessary.
  CodeGenFunction::OpaqueValueMapping binding(CGF, E);
  RegionCounter Cnt = CGF.getPGORegionCounter(E);

  Expr *condExpr = E->getCond();
  Expr *lhsExpr = E->getTrueExpr();
  Expr *rhsExpr = E->getFalseExpr();

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm.
  bool CondExprBool;
  if (CGF.ConstantFoldsToSimpleInteger(condExpr, CondExprBool)) {
    Expr *live = lhsExpr, *dead = rhsExpr;
    if (!CondExprBool) std::swap(live, dead);

    // If the dead side doesn't have labels we need, just emit the Live part.
    if (!CGF.ContainsLabel(dead)) {
      if (CondExprBool)
        Cnt.beginRegion(Builder);
      Value *Result = Visit(live);

      // If the live part is a throw expression, it acts like it has a void
      // type, so evaluating it returns a null Value*.  However, a conditional
      // with non-void type must return a non-null Value*.
      if (!Result && !E->getType()->isVoidType())
        Result = llvm::UndefValue::get(CGF.ConvertType(E->getType()));

      return Result;
    }
  }

  // OpenCL: If the condition is a vector, we can treat this condition like
  // the select function.
  if (CGF.getLangOpts().OpenCL
      && condExpr->getType()->isVectorType()) {
    Cnt.beginRegion(Builder);

    llvm::Value *CondV = CGF.EmitScalarExpr(condExpr);
    llvm::Value *LHS = Visit(lhsExpr);
    llvm::Value *RHS = Visit(rhsExpr);

    llvm::Type *condType = ConvertType(condExpr->getType());
    llvm::VectorType *vecTy = cast<llvm::VectorType>(condType);

    unsigned numElem = vecTy->getNumElements();
    llvm::Type *elemType = vecTy->getElementType();

    llvm::Value *zeroVec = llvm::Constant::getNullValue(vecTy);
    llvm::Value *TestMSB = Builder.CreateICmpSLT(CondV, zeroVec);
    llvm::Value *tmp = Builder.CreateSExt(TestMSB,
                                          llvm::VectorType::get(elemType,
                                                                numElem),
                                          "sext");
    llvm::Value *tmp2 = Builder.CreateNot(tmp);

    // Cast float to int to perform ANDs if necessary.
    llvm::Value *RHSTmp = RHS;
    llvm::Value *LHSTmp = LHS;
    bool wasCast = false;
    llvm::VectorType *rhsVTy = cast<llvm::VectorType>(RHS->getType());
    if (rhsVTy->getElementType()->isFloatingPointTy()) {
      RHSTmp = Builder.CreateBitCast(RHS, tmp2->getType());
      LHSTmp = Builder.CreateBitCast(LHS, tmp->getType());
      wasCast = true;
    }

    llvm::Value *tmp3 = Builder.CreateAnd(RHSTmp, tmp2);
    llvm::Value *tmp4 = Builder.CreateAnd(LHSTmp, tmp);
    llvm::Value *tmp5 = Builder.CreateOr(tmp3, tmp4, "cond");
    if (wasCast)
      tmp5 = Builder.CreateBitCast(tmp5, RHS->getType());

    return tmp5;
  }

  // If this is a really simple expression (like x ? 4 : 5), emit this as a
  // select instead of as control flow.  We can only do this if it is cheap and
  // safe to evaluate the LHS and RHS unconditionally.
  if (isCheapEnoughToEvaluateUnconditionally(lhsExpr, CGF) &&
      isCheapEnoughToEvaluateUnconditionally(rhsExpr, CGF)) {
    Cnt.beginRegion(Builder);

    llvm::Value *CondV = CGF.EvaluateExprAsBool(condExpr);
    llvm::Value *LHS = Visit(lhsExpr);
    llvm::Value *RHS = Visit(rhsExpr);
    if (!LHS) {
      // If the conditional has void type, make sure we return a null Value*.
      assert(!RHS && "LHS and RHS types must match");
      return nullptr;
    }
    return Builder.CreateSelect(CondV, LHS, RHS, "cond");
  }

  llvm::BasicBlock *LHSBlock = CGF.createBasicBlock("cond.true");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("cond.false");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("cond.end");

  CodeGenFunction::ConditionalEvaluation eval(CGF);
  CGF.EmitBranchOnBoolExpr(condExpr, LHSBlock, RHSBlock, Cnt.getCount());

  CGF.EmitBlock(LHSBlock);
  Cnt.beginRegion(Builder);
  eval.begin(CGF);
  Value *LHS = Visit(lhsExpr);
  eval.end(CGF);

  LHSBlock = Builder.GetInsertBlock();
  Builder.CreateBr(ContBlock);

  CGF.EmitBlock(RHSBlock);
  eval.begin(CGF);
  Value *RHS = Visit(rhsExpr);
  eval.end(CGF);

  RHSBlock = Builder.GetInsertBlock();
  CGF.EmitBlock(ContBlock);

  // If the LHS or RHS is a throw expression, it will be legitimately null.
  if (!LHS)
    return RHS;
  if (!RHS)
    return LHS;

  // Create a PHI node for the real part.
  llvm::PHINode *PN = Builder.CreatePHI(LHS->getType(), 2, "cond");
  PN->addIncoming(LHS, LHSBlock);
  PN->addIncoming(RHS, RHSBlock);
  return PN;
}

Value *ScalarExprEmitter::VisitChooseExpr(ChooseExpr *E) {
  return Visit(E->getChosenSubExpr());
}

Value *ScalarExprEmitter::VisitVAArgExpr(VAArgExpr *VE) {
  QualType Ty = VE->getType();

  if (Ty->isVariablyModifiedType())
    CGF.EmitVariablyModifiedType(Ty);

  llvm::Value *ArgValue = CGF.EmitVAListRef(VE->getSubExpr());
  llvm::Value *ArgPtr = CGF.EmitVAArg(ArgValue, VE->getType());
  llvm::Type *ArgTy = ConvertType(VE->getType());

  // If EmitVAArg fails, we fall back to the LLVM instruction.
  if (!ArgPtr)
    return Builder.CreateVAArg(ArgValue, ArgTy);

  // FIXME Volatility.
  llvm::Value *Val = Builder.CreateLoad(ArgPtr);

  // If EmitVAArg promoted the type, we must truncate it.
  if (ArgTy != Val->getType()) {
    if (ArgTy->isPointerTy() && !Val->getType()->isPointerTy())
      Val = Builder.CreateIntToPtr(Val, ArgTy);
    else
      Val = Builder.CreateTrunc(Val, ArgTy);
  }

  return Val;
}

Value *ScalarExprEmitter::VisitBlockExpr(const BlockExpr *block) {
  return CGF.EmitBlockLiteral(block);
}

Value *ScalarExprEmitter::VisitAsTypeExpr(AsTypeExpr *E) {
  Value *Src  = CGF.EmitScalarExpr(E->getSrcExpr());
  llvm::Type *DstTy = ConvertType(E->getType());

  // Going from vec4->vec3 or vec3->vec4 is a special case and requires
  // a shuffle vector instead of a bitcast.
  llvm::Type *SrcTy = Src->getType();
  if (isa<llvm::VectorType>(DstTy) && isa<llvm::VectorType>(SrcTy)) {
    unsigned numElementsDst = cast<llvm::VectorType>(DstTy)->getNumElements();
    unsigned numElementsSrc = cast<llvm::VectorType>(SrcTy)->getNumElements();
    if ((numElementsDst == 3 && numElementsSrc == 4)
        || (numElementsDst == 4 && numElementsSrc == 3)) {


      // In the case of going from int4->float3, a bitcast is needed before
      // doing a shuffle.
      llvm::Type *srcElemTy =
      cast<llvm::VectorType>(SrcTy)->getElementType();
      llvm::Type *dstElemTy =
      cast<llvm::VectorType>(DstTy)->getElementType();

      if ((srcElemTy->isIntegerTy() && dstElemTy->isFloatTy())
          || (srcElemTy->isFloatTy() && dstElemTy->isIntegerTy())) {
        // Create a float type of the same size as the source or destination.
        llvm::VectorType *newSrcTy = llvm::VectorType::get(dstElemTy,
                                                                 numElementsSrc);

        Src = Builder.CreateBitCast(Src, newSrcTy, "astypeCast");
      }

      llvm::Value *UnV = llvm::UndefValue::get(Src->getType());

      SmallVector<llvm::Constant*, 3> Args;
      Args.push_back(Builder.getInt32(0));
      Args.push_back(Builder.getInt32(1));
      Args.push_back(Builder.getInt32(2));

      if (numElementsDst == 4)
        Args.push_back(llvm::UndefValue::get(CGF.Int32Ty));

      llvm::Constant *Mask = llvm::ConstantVector::get(Args);

      return Builder.CreateShuffleVector(Src, UnV, Mask, "astype");
    }
  }

  return Builder.CreateBitCast(Src, DstTy, "astype");
}

Value *ScalarExprEmitter::VisitAtomicExpr(AtomicExpr *E) {
  return CGF.EmitAtomicExpr(E).getScalarVal();
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitScalarExpr - Emit the computation of the specified expression of scalar
/// type, ignoring the result.
Value *CodeGenFunction::EmitScalarExpr(const Expr *E, bool IgnoreResultAssign) {
  assert(E && hasScalarEvaluationKind(E->getType()) &&
         "Invalid scalar expression to emit");

  bool hasDebugInfo = getDebugInfo();
  if (isa<CXXDefaultArgExpr>(E))
    disableDebugInfo();
  Value *V = ScalarExprEmitter(*this, IgnoreResultAssign)
    .Visit(const_cast<Expr*>(E));
  if (isa<CXXDefaultArgExpr>(E) && hasDebugInfo)
    enableDebugInfo();
  return V;
}

/// EmitScalarConversion - Emit a conversion from the specified type to the
/// specified destination type, both of which are LLVM scalar types.
Value *CodeGenFunction::EmitScalarConversion(Value *Src, QualType SrcTy,
                                             QualType DstTy) {
  assert(hasScalarEvaluationKind(SrcTy) && hasScalarEvaluationKind(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this).EmitScalarConversion(Src, SrcTy, DstTy);
}

/// EmitComplexToScalarConversion - Emit a conversion from the specified complex
/// type to the specified destination type, where the destination type is an
/// LLVM scalar type.
Value *CodeGenFunction::EmitComplexToScalarConversion(ComplexPairTy Src,
                                                      QualType SrcTy,
                                                      QualType DstTy) {
  assert(SrcTy->isAnyComplexType() && hasScalarEvaluationKind(DstTy) &&
         "Invalid complex -> scalar conversion");
  return ScalarExprEmitter(*this).EmitComplexToScalarConversion(Src, SrcTy,
                                                                DstTy);
}


llvm::Value *CodeGenFunction::
EmitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                        bool isInc, bool isPre) {
  return ScalarExprEmitter(*this).EmitScalarPrePostIncDec(E, LV, isInc, isPre);
}

LValue CodeGenFunction::EmitObjCIsaExpr(const ObjCIsaExpr *E) {
  llvm::Value *V;
  // object->isa or (*object).isa
  // Generate code as for: *(Class*)object
  // build Class* type
  llvm::Type *ClassPtrTy = ConvertType(E->getType());

  Expr *BaseExpr = E->getBase();
  if (BaseExpr->isRValue()) {
    V = CreateMemTemp(E->getType(), "resval");
    llvm::Value *Src = EmitScalarExpr(BaseExpr);
    Builder.CreateStore(Src, V);
    V = ScalarExprEmitter(*this).EmitLoadOfLValue(
      MakeNaturalAlignAddrLValue(V, E->getType()), E->getExprLoc());
  } else {
    if (E->isArrow())
      V = ScalarExprEmitter(*this).EmitLoadOfLValue(BaseExpr);
    else
      V = EmitLValue(BaseExpr).getAddress();
  }

  // build Class* type
  ClassPtrTy = ClassPtrTy->getPointerTo();
  V = Builder.CreateBitCast(V, ClassPtrTy);
  return MakeNaturalAlignAddrLValue(V, E->getType());
}


LValue CodeGenFunction::EmitCompoundAssignmentLValue(
                                            const CompoundAssignOperator *E) {
  ScalarExprEmitter Scalar(*this);
  Value *Result = nullptr;
  switch (E->getOpcode()) {
#define COMPOUND_OP(Op)                                                       \
    case BO_##Op##Assign:                                                     \
      return Scalar.EmitCompoundAssignLValue(E, &ScalarExprEmitter::Emit##Op, \
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
