//===--- CIRGenExprAgg.cpp - Emit CIR Code from Aggregate Expressions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Aggregate Expr nodes as CIR code.
//
//===----------------------------------------------------------------------===//
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "UnimplementedFeatureGuarding.h"
#include "mlir/IR/Attributes.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace cir;
using namespace clang;

namespace {

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
/// Is the value of the given expression possibly a reference to or
/// into a __block variable?
static bool isBlockVarRef(const Expr *E) {
  // Make sure we look through parens.
  E = E->IgnoreParens();

  // Check for a direct reference to a __block variable.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *var = dyn_cast<VarDecl>(DRE->getDecl());
    return (var && var->hasAttr<BlocksAttr>());
  }

  // More complicated stuff.

  // Binary operators.
  if (const BinaryOperator *op = dyn_cast<BinaryOperator>(E)) {
    // For an assignment or pointer-to-member operation, just care
    // about the LHS.
    if (op->isAssignmentOp() || op->isPtrMemOp())
      return isBlockVarRef(op->getLHS());

    // For a comma, just care about the RHS.
    if (op->getOpcode() == BO_Comma)
      return isBlockVarRef(op->getRHS());

    // FIXME: pointer arithmetic?
    return false;

    // Check both sides of a conditional operator.
  } else if (const AbstractConditionalOperator *op =
                 dyn_cast<AbstractConditionalOperator>(E)) {
    return isBlockVarRef(op->getTrueExpr()) ||
           isBlockVarRef(op->getFalseExpr());

    // OVEs are required to support BinaryConditionalOperators.
  } else if (const OpaqueValueExpr *op = dyn_cast<OpaqueValueExpr>(E)) {
    if (const Expr *src = op->getSourceExpr())
      return isBlockVarRef(src);

    // Casts are necessary to get things like (*(int*)&var) = foo().
    // We don't really care about the kind of cast here, except
    // we don't want to look through l2r casts, because it's okay
    // to get the *value* in a __block variable.
  } else if (const CastExpr *cast = dyn_cast<CastExpr>(E)) {
    if (cast->getCastKind() == CK_LValueToRValue)
      return false;
    return isBlockVarRef(cast->getSubExpr());

    // Handle unary operators.  Again, just aggressively look through
    // it, ignoring the operation.
  } else if (const UnaryOperator *uop = dyn_cast<UnaryOperator>(E)) {
    return isBlockVarRef(uop->getSubExpr());

    // Look into the base of a field access.
  } else if (const MemberExpr *mem = dyn_cast<MemberExpr>(E)) {
    return isBlockVarRef(mem->getBase());

    // Look into the base of a subscript.
  } else if (const ArraySubscriptExpr *sub = dyn_cast<ArraySubscriptExpr>(E)) {
    return isBlockVarRef(sub->getBase());
  }

  return false;
}

class AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CIRGenFunction &CGF;
  AggValueSlot Dest;
  bool IsResultUnused;

  // Calls `Fn` with a valid return value slot, potentially creating a temporary
  // to do so. If a temporary is created, an appropriate copy into `Dest` will
  // be emitted, as will lifetime markers.
  //
  // The given function should take a ReturnValueSlot, and return an RValue that
  // points to said slot.
  void withReturnValueSlot(const Expr *E,
                           llvm::function_ref<RValue(ReturnValueSlot)> Fn);

  AggValueSlot EnsureSlot(mlir::Location loc, QualType T) {
    if (!Dest.isIgnored())
      return Dest;
    return CGF.CreateAggTemp(T, loc, "agg.tmp.ensured");
  }

  void EnsureDest(mlir::Location loc, QualType T) {
    if (!Dest.isIgnored())
      return;
    Dest = CGF.CreateAggTemp(T, loc, "agg.tmp.ensured");
  }

public:
  AggExprEmitter(CIRGenFunction &cgf, AggValueSlot Dest, bool IsResultUnused)
      : CGF{cgf}, Dest(Dest), IsResultUnused(IsResultUnused) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// Given an expression with aggregate type that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result into
  /// DestPtr.
  void buildAggLoadOfLValue(const Expr *E);

  enum ExprValueKind { EVK_RValue, EVK_NonRValue };

  /// Perform the final copy to DestPtr, if desired.
  void buildFinalDestCopy(QualType type, RValue src);

  /// Perform the final copy to DestPtr, if desired. SrcIsRValue is true if
  /// source comes from an RValue.
  void buildFinalDestCopy(QualType type, const LValue &src,
                          ExprValueKind SrcValueKind = EVK_NonRValue);
  void buildCopy(QualType type, const AggValueSlot &dest,
                 const AggValueSlot &src);

  void buildArrayInit(Address DestPtr, mlir::cir::ArrayType AType,
                      QualType ArrayQTy, Expr *ExprToVisit,
                      ArrayRef<Expr *> Args, Expr *ArrayFiller);

  AggValueSlot::NeedsGCBarriers_t needsGC(QualType T) {
    if (CGF.getLangOpts().getGC() && TypeRequiresGCollection(T))
      llvm_unreachable("garbage collection is NYI");
    return AggValueSlot::DoesNotNeedGCBarriers;
  }

  bool TypeRequiresGCollection(QualType T);

  //===--------------------------------------------------------------------===//
  //                             Visitor Methods
  //===--------------------------------------------------------------------===//

  void Visit(Expr *E) {
    if (CGF.getDebugInfo()) {
      llvm_unreachable("NYI");
    }
    StmtVisitor<AggExprEmitter>::Visit(E);
  }

  void VisitStmt(Stmt *S) {
    llvm::errs() << "Missing visitor for AggExprEmitter Stmt: "
                 << S->getStmtClassName() << "\n";
    llvm_unreachable("NYI");
  }
  void VisitParenExpr(ParenExpr *PE) { Visit(PE->getSubExpr()); }
  void VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    llvm_unreachable("NYI");
  }
  void VisitCoawaitExpr(CoawaitExpr *E) {
    CGF.buildCoawaitExpr(*E, Dest, IsResultUnused);
  }
  void VisitCoyieldExpr(CoyieldExpr *E) { llvm_unreachable("NYI"); }
  void VisitUnaryCoawait(UnaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitUnaryExtension(UnaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitConstantExpr(ConstantExpr *E) { llvm_unreachable("NYI"); }

  // l-values
  void VisitDeclRefExpr(DeclRefExpr *E) { buildAggLoadOfLValue(E); }
  void VisitMemberExpr(MemberExpr *E) { buildAggLoadOfLValue(E); }
  void VisitUnaryDeref(UnaryOperator *E) { buildAggLoadOfLValue(E); }
  void VisitStringLiteral(StringLiteral *E) { llvm_unreachable("NYI"); }
  void VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    buildAggLoadOfLValue(E);
  }
  void VisitPredefinedExpr(const PredefinedExpr *E) { llvm_unreachable("NYI"); }

  // Operators.
  void VisitCastExpr(CastExpr *E);
  void VisitCallExpr(const CallExpr *E);

  void VisitStmtExpr(const StmtExpr *E) {
    assert(!UnimplementedFeature::stmtExprEvaluation() && "NYI");
    CGF.buildCompoundStmt(*E->getSubStmt(), /*getLast=*/true, Dest);
  }

  void VisitBinaryOperator(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitPointerToDataMemberBinaryOperator(const BinaryOperator *E) {
    llvm_unreachable("NYI");
  }
  void VisitBinAssign(const BinaryOperator *E) {

    // For an assignment to work, the value on the right has
    // to be compatible with the value on the left.
    assert(CGF.getContext().hasSameUnqualifiedType(E->getLHS()->getType(),
                                                   E->getRHS()->getType()) &&
           "Invalid assignment");

    if (isBlockVarRef(E->getLHS()) &&
        E->getRHS()->HasSideEffects(CGF.getContext())) {
      llvm_unreachable("NYI");
    }

    LValue lhs = CGF.buildLValue(E->getLHS());

    // If we have an atomic type, evaluate into the destination and then
    // do an atomic copy.
    if (lhs.getType()->isAtomicType() ||
        CGF.LValueIsSuitableForInlineAtomic(lhs)) {
      assert(!UnimplementedFeature::atomicTypes());
      return;
    }

    // Codegen the RHS so that it stores directly into the LHS.
    AggValueSlot lhsSlot = AggValueSlot::forLValue(
        lhs, AggValueSlot::IsDestructed, AggValueSlot::DoesNotNeedGCBarriers,
        AggValueSlot::IsAliased, AggValueSlot::MayOverlap);

    // A non-volatile aggregate destination might have volatile member.
    if (!lhsSlot.isVolatile() && CGF.hasVolatileMember(E->getLHS()->getType()))
      assert(!UnimplementedFeature::atomicTypes());

    CGF.buildAggExpr(E->getRHS(), lhsSlot);

    // Copy into the destination if the assignment isn't ignored.
    buildFinalDestCopy(E->getType(), lhs);

    if (!Dest.isIgnored() && !Dest.isExternallyDestructed() &&
        E->getType().isDestructedType() == QualType::DK_nontrivial_c_struct)
      CGF.pushDestroy(QualType::DK_nontrivial_c_struct, Dest.getAddress(),
                      E->getType());
  }

  void VisitBinComma(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitBinCmp(const BinaryOperator *E);
  void VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *E) {
    llvm_unreachable("NYI");
  }

  void VisitObjCMessageExpr(ObjCMessageExpr *E) { llvm_unreachable("NYI"); }
  void VisitObjCIVarRefExpr(ObjCIvarRefExpr *E) { llvm_unreachable("NYI"); }

  void VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    llvm_unreachable("NYI");
  }
  void VisitChooseExpr(const ChooseExpr *E) { llvm_unreachable("NYI"); }
  void VisitInitListExpr(InitListExpr *E);
  void VisitCXXParenListOrInitListExpr(Expr *ExprToVisit, ArrayRef<Expr *> Args,
                                       FieldDecl *InitializedFieldInUnion,
                                       Expr *ArrayFiller);
  void VisitArrayInitLoopExpr(const ArrayInitLoopExpr *E,
                              llvm::Value *outerBegin = nullptr) {
    llvm_unreachable("NYI");
  }
  void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitNoInitExpr(NoInitExpr *E) { llvm_unreachable("NYI"); }
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    CIRGenFunction::CXXDefaultArgExprScope Scope(CGF, DAE);
    Visit(DAE->getExpr());
  }
  void VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    CIRGenFunction::CXXDefaultInitExprScope Scope(CGF, DIE);
    Visit(DIE->getExpr());
  }
  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);
  void VisitCXXConstructExpr(const CXXConstructExpr *E);
  void VisitCXXInheritedCtorInitExpr(const CXXInheritedCtorInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitLambdaExpr(LambdaExpr *E);
  void VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitExprWithCleanups(ExprWithCleanups *E);
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitCXXTypeidExpr(CXXTypeidExpr *E) { llvm_unreachable("NYI"); }
  void VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E);
  void VisitOpaqueValueExpr(OpaqueValueExpr *E) { llvm_unreachable("NYI"); }

  void VisitPseudoObjectExpr(PseudoObjectExpr *E) { llvm_unreachable("NYI"); }

  void VisitVAArgExpr(VAArgExpr *E) { llvm_unreachable("NYI"); }

  void buildInitializationToLValue(Expr *E, LValue LV);

  void buildNullInitializationToLValue(mlir::Location loc, LValue Address);
  void VisitCXXThrowExpr(const CXXThrowExpr *E) { llvm_unreachable("NYI"); }
  void VisitAtomicExpr(AtomicExpr *E) { llvm_unreachable("NYI"); }
};
} // namespace

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// Given an expression with aggregate type that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result into DestPtr.
void AggExprEmitter::buildAggLoadOfLValue(const Expr *E) {
  LValue LV = CGF.buildLValue(E);

  // If the type of the l-value is atomic, then do an atomic load.
  if (LV.getType()->isAtomicType() || CGF.LValueIsSuitableForInlineAtomic(LV) ||
      UnimplementedFeature::atomicTypes())
    llvm_unreachable("atomic load is NYI");

  buildFinalDestCopy(E->getType(), LV);
}

/// Perform the final copy to DestPtr, if desired.
void AggExprEmitter::buildFinalDestCopy(QualType type, RValue src) {
  assert(src.isAggregate() && "value must be aggregate value!");
  LValue srcLV = CGF.makeAddrLValue(src.getAggregateAddress(), type);
  buildFinalDestCopy(type, srcLV, EVK_RValue);
}

/// Perform the final copy to DestPtr, if desired.
void AggExprEmitter::buildFinalDestCopy(QualType type, const LValue &src,
                                        ExprValueKind SrcValueKind) {
  // If Dest is ignored, then we're evaluating an aggregate expression
  // in a context that doesn't care about the result.  Note that loads
  // from volatile l-values force the existence of a non-ignored
  // destination.
  if (Dest.isIgnored())
    return;

  // Copy non-trivial C structs here.
  if (Dest.isVolatile())
    assert(!UnimplementedFeature::volatileTypes());

  if (SrcValueKind == EVK_RValue) {
    if (type.isNonTrivialToPrimitiveDestructiveMove() == QualType::PCK_Struct) {
      llvm_unreachable("move assignment/move ctor for rvalue is NYI");
    }
  } else {
    if (type.isNonTrivialToPrimitiveCopy() == QualType::PCK_Struct)
      llvm_unreachable("non-trivial primitive copy is NYI");
  }

  AggValueSlot srcAgg = AggValueSlot::forLValue(
      src, AggValueSlot::IsDestructed, needsGC(type), AggValueSlot::IsAliased,
      AggValueSlot::MayOverlap);
  buildCopy(type, Dest, srcAgg);
}

/// Perform a copy from the source into the destination.
///
/// \param type - the type of the aggregate being copied; qualifiers are
///   ignored
void AggExprEmitter::buildCopy(QualType type, const AggValueSlot &dest,
                               const AggValueSlot &src) {
  if (dest.requiresGCollection())
    llvm_unreachable("garbage collection is NYI");

  // If the result of the assignment is used, copy the LHS there also.
  // It's volatile if either side is.  Use the minimum alignment of
  // the two sides.
  LValue DestLV = CGF.makeAddrLValue(dest.getAddress(), type);
  LValue SrcLV = CGF.makeAddrLValue(src.getAddress(), type);
  if (dest.isVolatile() || src.isVolatile() ||
      UnimplementedFeature::volatileTypes())
    llvm_unreachable("volatile is NYI");
  CGF.buildAggregateCopy(DestLV, SrcLV, type, dest.mayOverlap(), false);
}

// FIXME(cir): This function could be shared with traditional LLVM codegen
/// Determine if E is a trivial array filler, that is, one that is
/// equivalent to zero-initialization.
static bool isTrivialFiller(Expr *E) {
  if (!E)
    return true;

  if (isa<ImplicitValueInitExpr>(E))
    return true;

  if (auto *ILE = dyn_cast<InitListExpr>(E)) {
    if (ILE->getNumInits())
      return false;
    return isTrivialFiller(ILE->getArrayFiller());
  }

  if (auto *Cons = dyn_cast_or_null<CXXConstructExpr>(E))
    return Cons->getConstructor()->isDefaultConstructor() &&
           Cons->getConstructor()->isTrivial();

  // FIXME: Are there other cases where we can avoid emitting an initializer?
  return false;
}

void AggExprEmitter::buildArrayInit(Address DestPtr, mlir::cir::ArrayType AType,
                                    QualType ArrayQTy, Expr *ExprToVisit,
                                    ArrayRef<Expr *> Args, Expr *ArrayFiller) {
  uint64_t NumInitElements = Args.size();

  uint64_t NumArrayElements = AType.getSize();
  assert(NumInitElements != 0 && "expected at least one initializaed value");
  assert(NumInitElements <= NumArrayElements);

  QualType elementType =
      CGF.getContext().getAsArrayType(ArrayQTy)->getElementType();
  QualType elementPtrType = CGF.getContext().getPointerType(elementType);

  auto cirElementType = CGF.convertType(elementType);
  auto cirElementPtrType = mlir::cir::PointerType::get(
      CGF.getBuilder().getContext(), cirElementType);
  auto loc = CGF.getLoc(ExprToVisit->getSourceRange());

  // Cast from cir.ptr<cir.array<elementType> to cir.ptr<elementType>
  auto begin = CGF.getBuilder().create<mlir::cir::CastOp>(
      loc, cirElementPtrType, mlir::cir::CastKind::array_to_ptrdecay,
      DestPtr.getPointer());

  CharUnits elementSize = CGF.getContext().getTypeSizeInChars(elementType);
  CharUnits elementAlign =
      DestPtr.getAlignment().alignmentOfArrayElement(elementSize);

  // Exception safety requires us to destroy all the
  // already-constructed members if an initializer throws.
  // For that, we'll need an EH cleanup.
  [[maybe_unused]] QualType::DestructionKind dtorKind =
      elementType.isDestructedType();
  [[maybe_unused]] Address endOfInit = Address::invalid();
  assert(!CGF.needsEHCleanup(dtorKind) && "destructed types NIY");

  // The 'current element to initialize'.  The invariants on this
  // variable are complicated.  Essentially, after each iteration of
  // the loop, it points to the last initialized element, except
  // that it points to the beginning of the array before any
  // elements have been initialized.
  mlir::Value element = begin;

  // Don't build the 'one' before the cycle to avoid
  // emmiting the redundant cir.const(1) instrs.
  mlir::Value one;

  // Emit the explicit initializers.
  for (uint64_t i = 0; i != NumInitElements; ++i) {
    if (i == 1)
      one = CGF.getBuilder().getConstInt(
          loc, CGF.PtrDiffTy.cast<mlir::cir::IntType>(), 1);

    // Advance to the next element.
    if (i > 0) {
      element = CGF.getBuilder().create<mlir::cir::PtrStrideOp>(
          loc, cirElementPtrType, element, one);

      // Tell the cleanup that it needs to destroy up to this
      // element.  TODO: some of these stores can be trivially
      // observed to be unnecessary.
      assert(!endOfInit.isValid() && "destructed types NIY");
    }

    LValue elementLV = CGF.makeAddrLValue(
        Address(element, cirElementType, elementAlign), elementType);
    buildInitializationToLValue(Args[i], elementLV);
  }

  // Check whether there's a non-trivial array-fill expression.
  bool hasTrivialFiller = isTrivialFiller(ArrayFiller);

  // Any remaining elements need to be zero-initialized, possibly
  // using the filler expression.  We can skip this if the we're
  // emitting to zeroed memory.
  if (NumInitElements != NumArrayElements &&
      !(Dest.isZeroed() && hasTrivialFiller &&
        CGF.getTypes().isZeroInitializable(elementType))) {

    // Use an actual loop.  This is basically
    //   do { *array++ = filler; } while (array != end);

    auto &builder = CGF.getBuilder();

    // Advance to the start of the rest of the array.
    if (NumInitElements) {
      auto one =
          builder.getConstInt(loc, CGF.PtrDiffTy.cast<mlir::cir::IntType>(), 1);
      element = builder.create<mlir::cir::PtrStrideOp>(loc, cirElementPtrType,
                                                       element, one);

      assert(!endOfInit.isValid() && "destructed types NIY");
    }

    // Allocate the temporary variable
    // to store the pointer to first unitialized element
    auto tmpAddr = CGF.CreateTempAlloca(
        cirElementPtrType, CGF.getPointerAlign(), loc, "arrayinit.temp");
    LValue tmpLV = CGF.makeAddrLValue(tmpAddr, elementPtrType);
    CGF.buildStoreThroughLValue(RValue::get(element), tmpLV);

    // Compute the end of array
    auto numArrayElementsConst = builder.getConstInt(
        loc, CGF.PtrDiffTy.cast<mlir::cir::IntType>(), NumArrayElements);
    mlir::Value end = builder.create<mlir::cir::PtrStrideOp>(
        loc, cirElementPtrType, begin, numArrayElementsConst);

    builder.createDoWhile(
        loc,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto currentElement = builder.createLoad(loc, tmpAddr);
          mlir::Type boolTy = CGF.getCIRType(CGF.getContext().BoolTy);
          auto cmp = builder.create<mlir::cir::CmpOp>(
              loc, boolTy, mlir::cir::CmpOpKind::ne, currentElement, end);
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto currentElement = builder.createLoad(loc, tmpAddr);

          if (UnimplementedFeature::cleanups())
            llvm_unreachable("NYI");

          // Emit the actual filler expression.
          LValue elementLV = CGF.makeAddrLValue(
              Address(currentElement, cirElementType, elementAlign),
              elementType);
          if (ArrayFiller)
            buildInitializationToLValue(ArrayFiller, elementLV);
          else
            buildNullInitializationToLValue(loc, elementLV);

          // Tell the EH cleanup that we finished with the last element.
          assert(!endOfInit.isValid() && "destructed types NIY");

          // Advance pointer and store them to temporary variable
          auto one = builder.getConstInt(
              loc, CGF.PtrDiffTy.cast<mlir::cir::IntType>(), 1);
          auto nextElement = builder.create<mlir::cir::PtrStrideOp>(
              loc, cirElementPtrType, currentElement, one);
          CGF.buildStoreThroughLValue(RValue::get(nextElement), tmpLV);

          builder.createYield(loc);
        });
  }

  // Leave the partial-array cleanup if we entered one.
  assert(!dtorKind && "destructed types NIY");
}

/// True if the given aggregate type requires special GC API calls.
bool AggExprEmitter::TypeRequiresGCollection(QualType T) {
  // Only record types have members that might require garbage collection.
  const RecordType *RecordTy = T->getAs<RecordType>();
  if (!RecordTy)
    return false;

  // Don't mess with non-trivial C++ types.
  RecordDecl *Record = RecordTy->getDecl();
  if (isa<CXXRecordDecl>(Record) &&
      (cast<CXXRecordDecl>(Record)->hasNonTrivialCopyConstructor() ||
       !cast<CXXRecordDecl>(Record)->hasTrivialDestructor()))
    return false;

  // Check whether the type has an object member.
  return Record->hasObjectMember();
}

//===----------------------------------------------------------------------===//
//                             Visitor Methods
//===----------------------------------------------------------------------===//

/// Determine whether the given cast kind is known to always convert values
/// with all zero bits in their value representation to values with all zero
/// bits in their value representation.
/// TODO(cir): this can be shared with LLVM codegen.
static bool castPreservesZero(const CastExpr *CE) {
  switch (CE->getCastKind()) {
  case CK_HLSLVectorTruncation:
  case CK_HLSLArrayRValue:
    llvm_unreachable("NYI");
    // No-ops.
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
  case CK_BitCast:
  case CK_ToUnion:
  case CK_ToVoid:
    // Conversions between (possibly-complex) integral, (possibly-complex)
    // floating-point, and bool.
  case CK_BooleanToSignedIntegral:
  case CK_FloatingCast:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexToIntegralComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingRealToComplex:
  case CK_FloatingToBoolean:
  case CK_FloatingToIntegral:
  case CK_IntegralCast:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexToFloatingComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralRealToComplex:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
    // Reinterpreting integers as pointers and vice versa.
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
    // Language extensions.
  case CK_VectorSplat:
  case CK_MatrixCast:
  case CK_NonAtomicToAtomic:
  case CK_AtomicToNonAtomic:
    return true;

  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_NullToMemberPointer:
  case CK_ReinterpretMemberPointer:
    // FIXME: ABI-dependent.
    return false;

  case CK_AnyPointerToBlockPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_CPointerToObjCPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_IntToOCLSampler:
  case CK_ZeroToOCLOpaqueType:
    // FIXME: Check these.
    return false;

  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToFloating:
  case CK_FixedPointToIntegral:
  case CK_FloatingToFixedPoint:
  case CK_IntegralToFixedPoint:
    // FIXME: Do all fixed-point types represent zero as all 0 bits?
    return false;

  case CK_AddressSpaceConversion:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_Dynamic:
  case CK_NullToPointer:
  case CK_PointerToBoolean:
    // FIXME: Preserves zeroes only if zero pointers and null pointers have the
    // same representation in all involved address spaces.
    return false;

  case CK_ARCConsumeObject:
  case CK_ARCExtendBlockObject:
  case CK_ARCProduceObject:
  case CK_ARCReclaimReturnedObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_BuiltinFnToFnPtr:
  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_LValueToRValue:
  case CK_LValueToRValueBitCast:
  case CK_UncheckedDerivedToBase:
    return false;
  }
  llvm_unreachable("Unhandled clang::CastKind enum");
}

/// If emitting this value will obviously just cause a store of
/// zero to memory, return true.  This can return false if uncertain, so it just
/// handles simple cases.
static bool isSimpleZero(const Expr *E, CIRGenFunction &CGF) {
  E = E->IgnoreParens();
  while (auto *CE = dyn_cast<CastExpr>(E)) {
    if (!castPreservesZero(CE))
      break;
    E = CE->getSubExpr()->IgnoreParens();
  }

  // 0
  if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(E))
    return IL->getValue() == 0;
  // +0.0
  if (const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(E))
    return FL->getValue().isPosZero();
  // int()
  if ((isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E)) &&
      CGF.getTypes().isZeroInitializable(E->getType()))
    return true;
  // (int*)0 - Null pointer expressions.
  if (const CastExpr *ICE = dyn_cast<CastExpr>(E)) {
    return ICE->getCastKind() == CK_NullToPointer &&
           CGF.getTypes().isPointerZeroInitializable(E->getType()) &&
           !E->HasSideEffects(CGF.getContext());
  }
  // '\0'
  if (const CharacterLiteral *CL = dyn_cast<CharacterLiteral>(E))
    return CL->getValue() == 0;

  // Otherwise, hard case: conservatively return false.
  return false;
}

void AggExprEmitter::buildNullInitializationToLValue(mlir::Location loc,
                                                     LValue lv) {
  QualType type = lv.getType();

  // If the destination slot is already zeroed out before the aggregate is
  // copied into it, we don't have to emit any zeros here.
  if (Dest.isZeroed() && CGF.getTypes().isZeroInitializable(type))
    return;

  if (CGF.hasScalarEvaluationKind(type)) {
    // For non-aggregates, we can store the appropriate null constant.
    auto null = CGF.CGM.buildNullConstant(type, loc);
    // Note that the following is not equivalent to
    // EmitStoreThroughBitfieldLValue for ARC types.
    if (lv.isBitField()) {
      mlir::Value result;
      CGF.buildStoreThroughBitfieldLValue(RValue::get(null), lv, result);
    } else {
      assert(lv.isSimple());
      CGF.buildStoreOfScalar(null, lv, /* isInitialization */ true);
    }
  } else {
    // There's a potential optimization opportunity in combining
    // memsets; that would be easy for arrays, but relatively
    // difficult for structures with the current code.
    CGF.buildNullInitialization(loc, lv.getAddress(), lv.getType());
  }
}

void AggExprEmitter::buildInitializationToLValue(Expr *E, LValue LV) {
  QualType type = LV.getType();
  // FIXME: Ignore result?
  // FIXME: Are initializers affected by volatile?
  if (Dest.isZeroed() && isSimpleZero(E, CGF)) {
    // TODO(cir): LLVM codegen considers 'storing "i32 0" to a zero'd memory
    // location is a noop'. Consider emitting the store to zero in CIR, as to
    // model the actual user behavior, we can have a pass to optimize this out
    // later.
    return;
  }

  if (isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E)) {
    auto loc = E->getSourceRange().isValid() ? CGF.getLoc(E->getSourceRange())
                                             : *CGF.currSrcLoc;
    return buildNullInitializationToLValue(loc, LV);
  } else if (isa<NoInitExpr>(E)) {
    // Do nothing.
    return;
  } else if (type->isReferenceType()) {
    RValue RV = CGF.buildReferenceBindingToExpr(E);
    return CGF.buildStoreThroughLValue(RV, LV);
  }

  switch (CGF.getEvaluationKind(type)) {
  case TEK_Complex:
    llvm_unreachable("NYI");
    return;
  case TEK_Aggregate:
    CGF.buildAggExpr(
        E, AggValueSlot::forLValue(LV, AggValueSlot::IsDestructed,
                                   AggValueSlot::DoesNotNeedGCBarriers,
                                   AggValueSlot::IsNotAliased,
                                   AggValueSlot::MayOverlap, Dest.isZeroed()));
    return;
  case TEK_Scalar:
    if (LV.isSimple()) {
      CGF.buildScalarInit(E, CGF.getLoc(E->getSourceRange()), LV);
    } else {
      CGF.buildStoreThroughLValue(RValue::get(CGF.buildScalarExpr(E)), LV);
    }
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void AggExprEmitter::VisitMaterializeTemporaryExpr(
    MaterializeTemporaryExpr *E) {
  Visit(E->getSubExpr());
}

void AggExprEmitter::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  AggValueSlot Slot = EnsureSlot(CGF.getLoc(E->getSourceRange()), E->getType());
  CGF.buildCXXConstructExpr(E, Slot);
}

void AggExprEmitter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  if (Dest.isPotentiallyAliased() && E->getType().isPODType(CGF.getContext())) {
    llvm_unreachable("NYI");
  }

  AggValueSlot Slot = EnsureSlot(CGF.getLoc(E->getSourceRange()), E->getType());

  // Block-scope compound literals are destroyed at the end of the enclosing
  // scope in C.
  bool Destruct =
      !CGF.getLangOpts().CPlusPlus && !Slot.isExternallyDestructed();
  if (Destruct)
    llvm_unreachable("NYI");

  llvm_unreachable("NYI");

  if (Destruct)
    if (QualType::DestructionKind DtorKind = E->getType().isDestructedType())
      llvm_unreachable("NYI");
}

void AggExprEmitter::VisitExprWithCleanups(ExprWithCleanups *E) {
  if (UnimplementedFeature::cleanups())
    llvm_unreachable("NYI");

  auto &builder = CGF.getBuilder();
  auto scopeLoc = CGF.getLoc(E->getSourceRange());
  [[maybe_unused]] auto scope = builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                              builder.getInsertionBlock()};
        Visit(E->getSubExpr());
      });
}

void AggExprEmitter::VisitLambdaExpr(LambdaExpr *E) {
  CIRGenFunction::SourceLocRAIIObject loc{CGF, CGF.getLoc(E->getSourceRange())};
  AggValueSlot Slot = EnsureSlot(CGF.getLoc(E->getSourceRange()), E->getType());
  LLVM_ATTRIBUTE_UNUSED LValue SlotLV =
      CGF.makeAddrLValue(Slot.getAddress(), E->getType());

  // We'll need to enter cleanup scopes in case any of the element initializers
  // throws an exception.
  if (UnimplementedFeature::cleanups())
    llvm_unreachable("NYI");
  mlir::Operation *CleanupDominator = nullptr;

  auto CurField = E->getLambdaClass()->field_begin();
  auto captureInfo = E->capture_begin();
  for (auto &captureInit : E->capture_inits()) {
    // Pick a name for the field.
    llvm::StringRef fieldName = CurField->getName();
    const LambdaCapture &capture = *captureInfo;
    if (capture.capturesVariable()) {
      assert(!CurField->isBitField() && "lambdas don't have bitfield members!");
      ValueDecl *v = capture.getCapturedVar();
      fieldName = v->getName();
      CGF.getCIRGenModule().LambdaFieldToName[*CurField] = fieldName;
    } else {
      llvm_unreachable("NYI");
    }

    // Emit initialization
    LValue LV =
        CGF.buildLValueForFieldInitialization(SlotLV, *CurField, fieldName);
    if (CurField->hasCapturedVLAType()) {
      llvm_unreachable("NYI");
    }

    buildInitializationToLValue(captureInit, LV);

    // Push a destructor if necessary.
    if (QualType::DestructionKind DtorKind =
            CurField->getType().isDestructedType()) {
      llvm_unreachable("NYI");
    }

    CurField++;
    captureInfo++;
  }

  // Deactivate all the partial cleanups in reverse order, which generally means
  // popping them.
  if (UnimplementedFeature::cleanups())
    llvm_unreachable("NYI");

  // Destroy the placeholder if we made one.
  if (CleanupDominator)
    CleanupDominator->erase();
}

void AggExprEmitter::VisitCastExpr(CastExpr *E) {
  if (const auto *ECE = dyn_cast<ExplicitCastExpr>(E))
    CGF.CGM.buildExplicitCastExprType(ECE, &CGF);
  switch (E->getCastKind()) {

  case CK_LValueToRValue:
    // If we're loading from a volatile type, force the destination
    // into existence.
    if (E->getSubExpr()->getType().isVolatileQualified() ||
        UnimplementedFeature::volatileTypes()) {
      llvm_unreachable("volatile is NYI");
    }
    [[fallthrough]];

  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
    assert(CGF.getContext().hasSameUnqualifiedType(E->getSubExpr()->getType(),
                                                   E->getType()) &&
           "Implicit cast types must be compatible");
    Visit(E->getSubExpr());
    break;

  case CK_LValueBitCast:
    llvm_unreachable("should not be emitting lvalue bitcast as rvalue");

  case CK_Dependent:
  case CK_BitCast:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_BooleanToSignedIntegral:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_BuiltinFnToFnPtr:
  case CK_ZeroToOCLOpaqueType:
  case CK_MatrixCast:

  case CK_IntToOCLSampler:
  case CK_FloatingToFixedPoint:
  case CK_FixedPointToFloating:
  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToIntegral:
  case CK_IntegralToFixedPoint:
    llvm::errs() << "cast '" << E->getCastKindName()
                 << "' invalid for aggregate types\n";
    llvm_unreachable("cast kind invalid for aggregate types");
  default: {
    llvm::errs() << "cast kind not implemented: '" << E->getCastKindName()
                 << "'\n";
    assert(0 && "not implemented");
    break;
  }
  }
}

void AggExprEmitter::VisitCallExpr(const CallExpr *E) {
  if (E->getCallReturnType(CGF.getContext())->isReferenceType()) {
    llvm_unreachable("NYI");
  }

  withReturnValueSlot(
      E, [&](ReturnValueSlot Slot) { return CGF.buildCallExpr(E, Slot); });
}

void AggExprEmitter::withReturnValueSlot(
    const Expr *E, llvm::function_ref<RValue(ReturnValueSlot)> EmitCall) {
  QualType RetTy = E->getType();
  bool RequiresDestruction =
      !Dest.isExternallyDestructed() &&
      RetTy.isDestructedType() == QualType::DK_nontrivial_c_struct;

  // If it makes no observable difference, save a memcpy + temporary.
  //
  // We need to always provide our own temporary if destruction is required.
  // Otherwise, EmitCall will emit its own, notice that it's "unused", and end
  // its lifetime before we have the chance to emit a proper destructor call.
  bool UseTemp = Dest.isPotentiallyAliased() || Dest.requiresGCollection() ||
                 (RequiresDestruction && !Dest.getAddress().isValid());

  Address RetAddr = Address::invalid();
  assert(!UnimplementedFeature::shouldEmitLifetimeMarkers() && "NYI");

  if (!UseTemp) {
    RetAddr = Dest.getAddress();
  } else {
    RetAddr = CGF.CreateMemTemp(RetTy, CGF.getLoc(E->getSourceRange()), "tmp",
                                &RetAddr);
    assert(!UnimplementedFeature::shouldEmitLifetimeMarkers() && "NYI");
  }

  RValue Src =
      EmitCall(ReturnValueSlot(RetAddr, Dest.isVolatile(), IsResultUnused,
                               Dest.isExternallyDestructed()));

  if (!UseTemp)
    return;

  assert(Dest.isIgnored() || Dest.getPointer() != Src.getAggregatePointer());
  buildFinalDestCopy(E->getType(), Src);

  if (!RequiresDestruction) {
    // If there's no dtor to run, the copy was the last use of our temporary.
    // Since we're not guaranteed to be in an ExprWithCleanups, clean up
    // eagerly.
    assert(!UnimplementedFeature::shouldEmitLifetimeMarkers() && "NYI");
  }
}

void AggExprEmitter::VisitBinCmp(const BinaryOperator *E) {
  assert(CGF.getContext().hasSameType(E->getLHS()->getType(),
                                      E->getRHS()->getType()));
  const ComparisonCategoryInfo &CmpInfo =
      CGF.getContext().CompCategories.getInfoForType(E->getType());
  assert(CmpInfo.Record->isTriviallyCopyable() &&
         "cannot copy non-trivially copyable aggregate");

  QualType ArgTy = E->getLHS()->getType();

  if (!ArgTy->isIntegralOrEnumerationType() && !ArgTy->isRealFloatingType() &&
      !ArgTy->isNullPtrType() && !ArgTy->isPointerType() &&
      !ArgTy->isMemberPointerType() && !ArgTy->isAnyComplexType())
    llvm_unreachable("aggregate three-way comparison");

  auto Loc = CGF.getLoc(E->getSourceRange());

  if (E->getType()->isAnyComplexType())
    llvm_unreachable("NYI");

  auto LHS = CGF.buildAnyExpr(E->getLHS()).getScalarVal();
  auto RHS = CGF.buildAnyExpr(E->getRHS()).getScalarVal();

  mlir::Value ResultScalar;
  if (ArgTy->isNullPtrType()) {
    ResultScalar =
        CGF.builder.getConstInt(Loc, CmpInfo.getEqualOrEquiv()->getIntValue());
  } else {
    auto LtRes = CmpInfo.getLess()->getIntValue();
    auto EqRes = CmpInfo.getEqualOrEquiv()->getIntValue();
    auto GtRes = CmpInfo.getGreater()->getIntValue();
    if (!CmpInfo.isPartial()) {
      // Strong ordering.
      ResultScalar = CGF.builder.createThreeWayCmpStrong(Loc, LHS, RHS, LtRes,
                                                         EqRes, GtRes);
    } else {
      // Partial ordering.
      auto UnorderedRes = CmpInfo.getUnordered()->getIntValue();
      ResultScalar = CGF.builder.createThreeWayCmpPartial(
          Loc, LHS, RHS, LtRes, EqRes, GtRes, UnorderedRes);
    }
  }

  // Create the return value in the destination slot.
  EnsureDest(Loc, E->getType());
  LValue DestLV = CGF.makeAddrLValue(Dest.getAddress(), E->getType());

  // Emit the address of the first (and only) field in the comparison category
  // type, and initialize it from the constant integer value produced above.
  const FieldDecl *ResultField = *CmpInfo.Record->field_begin();
  LValue FieldLV = CGF.buildLValueForFieldInitialization(
      DestLV, ResultField, ResultField->getName());
  CGF.buildStoreThroughLValue(RValue::get(ResultScalar), FieldLV);

  // All done! The result is in the Dest slot.
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *E) {
  // TODO(cir): use something like CGF.ErrorUnsupported
  if (E->hadArrayRangeDesignator())
    llvm_unreachable("GNU array range designator extension");

  if (E->isTransparent())
    return Visit(E->getInit(0));

  VisitCXXParenListOrInitListExpr(
      E, E->inits(), E->getInitializedFieldInUnion(), E->getArrayFiller());
}

void AggExprEmitter::VisitCXXParenListOrInitListExpr(
    Expr *ExprToVisit, ArrayRef<Expr *> InitExprs,
    FieldDecl *InitializedFieldInUnion, Expr *ArrayFiller) {
#if 0
  // FIXME: Assess perf here?  Figure out what cases are worth optimizing here
  // (Length of globals? Chunks of zeroed-out space?).
  //
  // If we can, prefer a copy from a global; this is a lot less code for long
  // globals, and it's easier for the current optimizers to analyze.
  if (llvm::Constant *C =
          CGF.CGM.EmitConstantExpr(ExprToVisit, ExprToVisit->getType(), &CGF)) {
    llvm::GlobalVariable* GV =
    new llvm::GlobalVariable(CGF.CGM.getModule(), C->getType(), true,
                             llvm::GlobalValue::InternalLinkage, C, "");
    EmitFinalDestCopy(ExprToVisit->getType(),
                      CGF.MakeAddrLValue(GV, ExprToVisit->getType()));
    return;
  }
#endif

  AggValueSlot Dest = EnsureSlot(CGF.getLoc(ExprToVisit->getSourceRange()),
                                 ExprToVisit->getType());

  LValue DestLV = CGF.makeAddrLValue(Dest.getAddress(), ExprToVisit->getType());

  // Handle initialization of an array.
  if (ExprToVisit->getType()->isConstantArrayType()) {
    auto AType = cast<mlir::cir::ArrayType>(Dest.getAddress().getElementType());
    buildArrayInit(Dest.getAddress(), AType, ExprToVisit->getType(),
                   ExprToVisit, InitExprs, ArrayFiller);
    return;
  } else if (ExprToVisit->getType()->isVariableArrayType()) {
    llvm_unreachable("variable arrays NYI");
    return;
  }

  if (ExprToVisit->getType()->isArrayType()) {
    llvm_unreachable("NYI");
  }

  assert(ExprToVisit->getType()->isRecordType() &&
         "Only support structs/unions here!");

  // Do struct initialization; this code just sets each individual member
  // to the approprate value.  This makes bitfield support automatic;
  // the disadvantage is that the generated code is more difficult for
  // the optimizer, especially with bitfields.
  unsigned NumInitElements = InitExprs.size();
  RecordDecl *record = ExprToVisit->getType()->castAs<RecordType>()->getDecl();

  // We'll need to enter cleanup scopes in case any of the element
  // initializers throws an exception.
  SmallVector<EHScopeStack::stable_iterator, 16> cleanups;
  // FIXME(cir): placeholder
  mlir::Operation *cleanupDominator = nullptr;
  [[maybe_unused]] auto addCleanup =
      [&](const EHScopeStack::stable_iterator &cleanup) {
        llvm_unreachable("NYI");
      };

  unsigned curInitIndex = 0;

  // Emit initialization of base classes.
  if (auto *CXXRD = dyn_cast<CXXRecordDecl>(record)) {
    assert(NumInitElements >= CXXRD->getNumBases() &&
           "missing initializer for base class");
    for ([[maybe_unused]] auto &Base : CXXRD->bases()) {
      llvm_unreachable("NYI");
    }
  }

  // Prepare a 'this' for CXXDefaultInitExprs.
  CIRGenFunction::FieldConstructionScope FCS(CGF, Dest.getAddress());

  if (record->isUnion()) {
    // Only initialize one field of a union. The field itself is
    // specified by the initializer list.
    if (!InitializedFieldInUnion) {
      // Empty union; we have nothing to do.

#ifndef NDEBUG
      // Make sure that it's really an empty and not a failure of
      // semantic analysis.
      for (const auto *Field : record->fields())
        assert(
            (Field->isUnnamedBitField() || Field->isAnonymousStructOrUnion()) &&
            "Only unnamed bitfields or ananymous class allowed");
#endif
      return;
    }

    // FIXME: volatility
    FieldDecl *Field = InitializedFieldInUnion;

    LValue FieldLoc =
        CGF.buildLValueForFieldInitialization(DestLV, Field, Field->getName());
    if (NumInitElements) {
      // Store the initializer into the field
      buildInitializationToLValue(InitExprs[0], FieldLoc);
    } else {
      // Default-initialize to null.
      buildNullInitializationToLValue(CGF.getLoc(ExprToVisit->getSourceRange()),
                                      FieldLoc);
    }

    return;
  }

  // Here we iterate over the fields; this makes it simpler to both
  // default-initialize fields and skip over unnamed fields.
  for (const auto *field : record->fields()) {
    // We're done once we hit the flexible array member.
    if (field->getType()->isIncompleteArrayType())
      break;

    // Always skip anonymous bitfields.
    if (field->isUnnamedBitField())
      continue;

    // We're done if we reach the end of the explicit initializers, we
    // have a zeroed object, and the rest of the fields are
    // zero-initializable.
    if (curInitIndex == NumInitElements && Dest.isZeroed() &&
        CGF.getTypes().isZeroInitializable(ExprToVisit->getType()))
      break;
    LValue LV =
        CGF.buildLValueForFieldInitialization(DestLV, field, field->getName());
    // We never generate write-barries for initialized fields.
    assert(!UnimplementedFeature::setNonGC());

    if (curInitIndex < NumInitElements) {
      // Store the initializer into the field.
      CIRGenFunction::SourceLocRAIIObject loc{
          CGF, CGF.getLoc(record->getSourceRange())};
      buildInitializationToLValue(InitExprs[curInitIndex++], LV);
    } else {
      // We're out of initializers; default-initialize to null
      buildNullInitializationToLValue(CGF.getLoc(ExprToVisit->getSourceRange()),
                                      LV);
    }

    // Push a destructor if necessary.
    // FIXME: if we have an array of structures, all explicitly
    // initialized, we can end up pushing a linear number of cleanups.
    [[maybe_unused]] bool pushedCleanup = false;
    if (QualType::DestructionKind dtorKind =
            field->getType().isDestructedType()) {
      llvm_unreachable("NYI");
    }

    // From LLVM codegen, maybe not useful for CIR:
    // If the GEP didn't get used because of a dead zero init or something
    // else, clean it up for -O0 builds and general tidiness.
  }

  // Deactivate all the partial cleanups in reverse order, which
  // generally means popping them.
  assert((cleanupDominator || cleanups.empty()) &&
         "Missing cleanupDominator before deactivating cleanup blocks");
  for (unsigned i = cleanups.size(); i != 0; --i)
    llvm_unreachable("NYI");

  // Destroy the placeholder if we made one.
  if (cleanupDominator)
    llvm_unreachable("NYI");
}

void AggExprEmitter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  // Ensure that we have a slot, but if we already do, remember
  // whether it was externally destructed.
  bool wasExternallyDestructed = Dest.isExternallyDestructed();
  EnsureDest(CGF.getLoc(E->getSourceRange()), E->getType());

  // We're going to push a destructor if there isn't already one.
  Dest.setExternallyDestructed();

  Visit(E->getSubExpr());

  // Push that destructor we promised.
  if (!wasExternallyDestructed)
    CGF.buildCXXTemporary(E->getTemporary(), E->getType(), Dest.getAddress());
}

//===----------------------------------------------------------------------===//
//                        Helpers and dispatcher
//===----------------------------------------------------------------------===//

/// Get an approximate count of the number of non-zero bytes that will be stored
/// when outputting the initializer for the specified initializer expression.
/// FIXME(cir): this can be shared with LLVM codegen.
static CharUnits GetNumNonZeroBytesInInit(const Expr *E, CIRGenFunction &CGF) {
  if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->getSubExpr();
  E = E->IgnoreParenNoopCasts(CGF.getContext());

  // 0 and 0.0 won't require any non-zero stores!
  if (isSimpleZero(E, CGF))
    return CharUnits::Zero();

  // If this is an initlist expr, sum up the size of sizes of the (present)
  // elements.  If this is something weird, assume the whole thing is non-zero.
  const InitListExpr *ILE = dyn_cast<InitListExpr>(E);
  while (ILE && ILE->isTransparent())
    ILE = dyn_cast<InitListExpr>(ILE->getInit(0));
  if (!ILE || !CGF.getTypes().isZeroInitializable(ILE->getType()))
    return CGF.getContext().getTypeSizeInChars(E->getType());

  // InitListExprs for structs have to be handled carefully.  If there are
  // reference members, we need to consider the size of the reference, not the
  // referencee.  InitListExprs for unions and arrays can't have references.
  if (const RecordType *RT = E->getType()->getAs<RecordType>()) {
    if (!RT->isUnionType()) {
      RecordDecl *SD = RT->getDecl();
      CharUnits NumNonZeroBytes = CharUnits::Zero();

      unsigned ILEElement = 0;
      if (auto *CXXRD = dyn_cast<CXXRecordDecl>(SD))
        while (ILEElement != CXXRD->getNumBases())
          NumNonZeroBytes +=
              GetNumNonZeroBytesInInit(ILE->getInit(ILEElement++), CGF);
      for (const auto *Field : SD->fields()) {
        // We're done once we hit the flexible array member or run out of
        // InitListExpr elements.
        if (Field->getType()->isIncompleteArrayType() ||
            ILEElement == ILE->getNumInits())
          break;
        if (Field->isUnnamedBitField())
          continue;

        const Expr *E = ILE->getInit(ILEElement++);

        // Reference values are always non-null and have the width of a pointer.
        if (Field->getType()->isReferenceType())
          NumNonZeroBytes += CGF.getContext().toCharUnitsFromBits(
              CGF.getTarget().getPointerWidth(LangAS::Default));
        else
          NumNonZeroBytes += GetNumNonZeroBytesInInit(E, CGF);
      }

      return NumNonZeroBytes;
    }
  }

  // FIXME: This overestimates the number of non-zero bytes for bit-fields.
  CharUnits NumNonZeroBytes = CharUnits::Zero();
  for (unsigned i = 0, e = ILE->getNumInits(); i != e; ++i)
    NumNonZeroBytes += GetNumNonZeroBytesInInit(ILE->getInit(i), CGF);
  return NumNonZeroBytes;
}

/// If the initializer is large and has a lot of zeros in it, emit a memset and
/// avoid storing the individual zeros.
static void CheckAggExprForMemSetUse(AggValueSlot &Slot, const Expr *E,
                                     CIRGenFunction &CGF) {
  // If the slot is arleady known to be zeroed, nothing to do. Don't mess with
  // volatile stores.
  if (Slot.isZeroed() || Slot.isVolatile() || !Slot.getAddress().isValid())
    return;

  // C++ objects with a user-declared constructor don't need zero'ing.
  if (CGF.getLangOpts().CPlusPlus)
    if (const auto *RT = CGF.getContext()
                             .getBaseElementType(E->getType())
                             ->getAs<RecordType>()) {
      const auto *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasUserDeclaredConstructor())
        return;
    }

  // If the type is 16-bytes or smaller, prefer individual stores over memset.
  CharUnits Size = Slot.getPreferredSize(CGF.getContext(), E->getType());
  if (Size <= CharUnits::fromQuantity(16))
    return;

  // Check to see if over 3/4 of the initializer are known to be zero.  If so,
  // we prefer to emit memset + individual stores for the rest.
  CharUnits NumNonZeroBytes = GetNumNonZeroBytesInInit(E, CGF);
  if (NumNonZeroBytes * 4 > Size)
    return;

  // Okay, it seems like a good idea to use an initial memset, emit the call.
  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(E->getSourceRange());
  Address slotAddr = Slot.getAddress();
  auto zero = builder.getZero(loc, slotAddr.getElementType());

  builder.createStore(loc, zero, slotAddr);
  // Loc = CGF.Builder.CreateElementBitCast(Loc, CGF.Int8Ty);
  // CGF.Builder.CreateMemSet(Loc, CGF.Builder.getInt8(0), SizeVal, false);

  // Tell the AggExprEmitter that the slot is known zero.
  Slot.setZeroed();
}

AggValueSlot::Overlap_t CIRGenFunction::getOverlapForBaseInit(
    const CXXRecordDecl *RD, const CXXRecordDecl *BaseRD, bool IsVirtual) {
  // If the most-derived object is a field declared with [[no_unique_address]],
  // the tail padding of any virtual base could be reused for other subobjects
  // of that field's class.
  if (IsVirtual)
    return AggValueSlot::MayOverlap;

  // If the base class is laid out entirely within the nvsize of the derived
  // class, its tail padding cannot yet be initialized, so we can issue
  // stores at the full width of the base class.
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
  if (Layout.getBaseClassOffset(BaseRD) +
          getContext().getASTRecordLayout(BaseRD).getSize() <=
      Layout.getNonVirtualSize())
    return AggValueSlot::DoesNotOverlap;

  // The tail padding may contain values we need to preserve.
  return AggValueSlot::MayOverlap;
}

void CIRGenFunction::buildAggExpr(const Expr *E, AggValueSlot Slot) {
  assert(E && CIRGenFunction::hasAggregateEvaluationKind(E->getType()) &&
         "Invalid aggregate expression to emit");
  assert((Slot.getAddress().isValid() || Slot.isIgnored()) &&
         "slot has bits but no address");

  // Optimize the slot if possible.
  CheckAggExprForMemSetUse(Slot, E, *this);

  AggExprEmitter(*this, Slot, Slot.isIgnored()).Visit(const_cast<Expr *>(E));
}

void CIRGenFunction::buildAggregateCopy(LValue Dest, LValue Src, QualType Ty,
                                        AggValueSlot::Overlap_t MayOverlap,
                                        bool isVolatile) {
  // TODO(cir): this function needs improvements, commented code for now since
  // this will be touched again soon.
  assert(!Ty->isAnyComplexType() && "Shouldn't happen for complex");

  Address DestPtr = Dest.getAddress();
  Address SrcPtr = Src.getAddress();

  if (getLangOpts().CPlusPlus) {
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      CXXRecordDecl *Record = cast<CXXRecordDecl>(RT->getDecl());
      assert((Record->hasTrivialCopyConstructor() ||
              Record->hasTrivialCopyAssignment() ||
              Record->hasTrivialMoveConstructor() ||
              Record->hasTrivialMoveAssignment() ||
              Record->hasAttr<TrivialABIAttr>() || Record->isUnion()) &&
             "Trying to aggregate-copy a type without a trivial copy/move "
             "constructor or assignment operator");
      // Ignore empty classes in C++.
      if (Record->isEmpty())
        return;
    }
  }

  if (getLangOpts().CUDAIsDevice) {
    llvm_unreachable("CUDA is NYI");
  }

  // Aggregate assignment turns into llvm.memcpy.  This is almost valid per
  // C99 6.5.16.1p3, which states "If the value being stored in an object is
  // read from another object that overlaps in anyway the storage of the first
  // object, then the overlap shall be exact and the two objects shall have
  // qualified or unqualified versions of a compatible type."
  //
  // memcpy is not defined if the source and destination pointers are exactly
  // equal, but other compilers do this optimization, and almost every memcpy
  // implementation handles this case safely.  If there is a libc that does not
  // safely handle this, we can add a target hook.

  // Get data size info for this aggregate. Don't copy the tail padding if this
  // might be a potentially-overlapping subobject, since the tail padding might
  // be occupied by a different object. Otherwise, copying it is fine.
  TypeInfoChars TypeInfo;
  if (MayOverlap)
    TypeInfo = getContext().getTypeInfoDataSizeInChars(Ty);
  else
    TypeInfo = getContext().getTypeInfoInChars(Ty);

  mlir::Attribute SizeVal = nullptr;
  if (TypeInfo.Width.isZero()) {
    // But note that getTypeInfo returns 0 for a VLA.
    if (auto *VAT = dyn_cast_or_null<VariableArrayType>(
            getContext().getAsArrayType(Ty))) {
      llvm_unreachable("VLA is NYI");
    }
  }
  if (!SizeVal) {
    // NOTE(cir): CIR types already carry info about their sizes. This is here
    // just for codegen parity.
    SizeVal = builder.getI64IntegerAttr(TypeInfo.Width.getQuantity());
  }

  // FIXME: If we have a volatile struct, the optimizer can remove what might
  // appear to be `extra' memory ops:
  //
  // volatile struct { int i; } a, b;
  //
  // int main() {
  //   a = b;
  //   a = b;
  // }
  //
  // we need to use a different call here.  We use isVolatile to indicate when
  // either the source or the destination is volatile.

  // NOTE(cir): original codegen would normally convert DestPtr and SrcPtr to
  // i8* since memcpy operates on bytes. We don't need that in CIR because
  // cir.copy will operate on any CIR pointer that points to a sized type.

  // Don't do any of the memmove_collectable tests if GC isn't set.
  if (CGM.getLangOpts().getGC() == LangOptions::NonGC) {
    // fall through
  } else if (const RecordType *RecordTy = Ty->getAs<RecordType>()) {
    RecordDecl *Record = RecordTy->getDecl();
    if (Record->hasObjectMember()) {
      llvm_unreachable("ObjC is NYI");
    }
  } else if (Ty->isArrayType()) {
    QualType BaseType = getContext().getBaseElementType(Ty);
    if (const RecordType *RecordTy = BaseType->getAs<RecordType>()) {
      if (RecordTy->getDecl()->hasObjectMember()) {
        llvm_unreachable("ObjC is NYI");
      }
    }
  }

  builder.createCopy(DestPtr.getPointer(), SrcPtr.getPointer());

  // Determine the metadata to describe the position of any padding in this
  // memcpy, as well as the TBAA tags for the members of the struct, in case
  // the optimizer wishes to expand it in to scalar memory operations.
  if (CGM.getCodeGenOpts().NewStructPathTBAA || UnimplementedFeature::tbaa())
    llvm_unreachable("TBAA is NYI");
}

AggValueSlot::Overlap_t
CIRGenFunction::getOverlapForFieldInit(const FieldDecl *FD) {
  if (!FD->hasAttr<NoUniqueAddressAttr>() || !FD->getType()->isRecordType())
    return AggValueSlot::DoesNotOverlap;

  // If the field lies entirely within the enclosing class's nvsize, its tail
  // padding cannot overlap any already-initialized object. (The only subobjects
  // with greater addresses that might already be initialized are vbases.)
  const RecordDecl *ClassRD = FD->getParent();
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(ClassRD);
  if (Layout.getFieldOffset(FD->getFieldIndex()) +
          getContext().getTypeSize(FD->getType()) <=
      (uint64_t)getContext().toBits(Layout.getNonVirtualSize()))
    return AggValueSlot::DoesNotOverlap;

  // The tail padding may contain values we need to preserve.
  return AggValueSlot::MayOverlap;
}
