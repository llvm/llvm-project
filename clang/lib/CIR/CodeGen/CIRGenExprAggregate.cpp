//===- CIRGenExprAggregrate.cpp - Emit CIR Code from Aggregate Expressions ===//
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

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenValue.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"

#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include <cstdint>

using namespace clang;
using namespace clang::CIRGen;

namespace {
class AggExprEmitter : public StmtVisitor<AggExprEmitter> {

  CIRGenFunction &cgf;
  AggValueSlot dest;

  // Calls `fn` with a valid return value slot, potentially creating a temporary
  // to do so. If a temporary is created, an appropriate copy into `Dest` will
  // be emitted, as will lifetime markers.
  //
  // The given function should take a ReturnValueSlot, and return an RValue that
  // points to said slot.
  void withReturnValueSlot(const Expr *e,
                           llvm::function_ref<RValue(ReturnValueSlot)> fn);

  AggValueSlot ensureSlot(mlir::Location loc, QualType t) {
    if (!dest.isIgnored())
      return dest;

    cgf.cgm.errorNYI(loc, "Slot for ignored address");
    return dest;
  }

public:
  AggExprEmitter(CIRGenFunction &cgf, AggValueSlot dest)
      : cgf(cgf), dest(dest) {}

  /// Given an expression with aggregate type that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result into
  /// DestPtr.
  void emitAggLoadOfLValue(const Expr *e);

  void emitArrayInit(Address destPtr, cir::ArrayType arrayTy, QualType arrayQTy,
                     Expr *exprToVisit, ArrayRef<Expr *> args,
                     Expr *arrayFiller);

  /// Perform the final copy to DestPtr, if desired.
  void emitFinalDestCopy(QualType type, const LValue &src);

  void emitInitializationToLValue(Expr *e, LValue lv);

  void emitNullInitializationToLValue(mlir::Location loc, LValue lv);

  void Visit(Expr *e) { StmtVisitor<AggExprEmitter>::Visit(e); }

  void VisitCallExpr(const CallExpr *e);
  void VisitStmtExpr(const StmtExpr *e) {
    CIRGenFunction::StmtExprEvaluation eval(cgf);
    Address retAlloca =
        cgf.createMemTemp(e->getType(), cgf.getLoc(e->getSourceRange()));
    (void)cgf.emitCompoundStmt(*e->getSubStmt(), &retAlloca, dest);
  }

  void VisitDeclRefExpr(DeclRefExpr *e) { emitAggLoadOfLValue(e); }

  void VisitInitListExpr(InitListExpr *e);
  void VisitCXXConstructExpr(const CXXConstructExpr *e);

  void visitCXXParenListOrInitListExpr(Expr *e, ArrayRef<Expr *> args,
                                       FieldDecl *initializedFieldInUnion,
                                       Expr *arrayFiller);

  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *e) {
    assert(!cir::MissingFeatures::aggValueSlotDestructedFlag());
    Visit(e->getSubExpr());
  }

  // Stubs -- These should be moved up when they are implemented.
  void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *e) {
    // We shouldn't really get here, but we do because of missing handling for
    // emitting constant aggregate initializers. If we just ignore this, a
    // fallback handler will do the right thing.
    assert(!cir::MissingFeatures::constEmitterAggILE());
    return;
  }
  void VisitCastExpr(CastExpr *e) {
    switch (e->getCastKind()) {
    case CK_LValueToRValue:
      assert(!cir::MissingFeatures::aggValueSlotVolatile());
      [[fallthrough]];
    case CK_NoOp:
    case CK_UserDefinedConversion:
    case CK_ConstructorConversion:
      assert(cgf.getContext().hasSameUnqualifiedType(e->getSubExpr()->getType(),
                                                     e->getType()) &&
             "Implicit cast types must be compatible");
      Visit(e->getSubExpr());
      break;
    default:
      cgf.cgm.errorNYI(e->getSourceRange(),
                       std::string("AggExprEmitter: VisitCastExpr: ") +
                           e->getCastKindName());
      break;
    }
  }
  void VisitStmt(Stmt *s) {
    cgf.cgm.errorNYI(s->getSourceRange(),
                     std::string("AggExprEmitter::VisitStmt: ") +
                         s->getStmtClassName());
  }
  void VisitParenExpr(ParenExpr *pe) {
    cgf.cgm.errorNYI(pe->getSourceRange(), "AggExprEmitter: VisitParenExpr");
  }
  void VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    cgf.cgm.errorNYI(ge->getSourceRange(),
                     "AggExprEmitter: VisitGenericSelectionExpr");
  }
  void VisitCoawaitExpr(CoawaitExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitCoawaitExpr");
  }
  void VisitCoyieldExpr(CoyieldExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitCoyieldExpr");
  }
  void VisitUnaryCoawait(UnaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitUnaryCoawait");
  }
  void VisitUnaryExtension(UnaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitUnaryExtension");
  }
  void VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitSubstNonTypeTemplateParmExpr");
  }
  void VisitConstantExpr(ConstantExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitConstantExpr");
  }
  void VisitMemberExpr(MemberExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitMemberExpr");
  }
  void VisitUnaryDeref(UnaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitUnaryDeref");
  }
  void VisitStringLiteral(StringLiteral *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitStringLiteral");
  }
  void VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitCompoundLiteralExpr");
  }
  void VisitArraySubscriptExpr(ArraySubscriptExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitArraySubscriptExpr");
  }
  void VisitPredefinedExpr(const PredefinedExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitPredefinedExpr");
  }
  void VisitBinaryOperator(const BinaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitBinaryOperator");
  }
  void VisitPointerToDataMemberBinaryOperator(const BinaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitPointerToDataMemberBinaryOperator");
  }
  void VisitBinAssign(const BinaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitBinAssign");
  }
  void VisitBinComma(const BinaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitBinComma");
  }
  void VisitBinCmp(const BinaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitBinCmp");
  }
  void VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitCXXRewrittenBinaryOperator");
  }
  void VisitObjCMessageExpr(ObjCMessageExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitObjCMessageExpr");
  }
  void VisitObjCIVarRefExpr(ObjCIvarRefExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitObjCIVarRefExpr");
  }

  void VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitDesignatedInitUpdateExpr");
  }
  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitAbstractConditionalOperator");
  }
  void VisitChooseExpr(const ChooseExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitChooseExpr");
  }
  void VisitCXXParenListInitExpr(CXXParenListInitExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitCXXParenListInitExpr");
  }
  void VisitArrayInitLoopExpr(const ArrayInitLoopExpr *e,
                              llvm::Value *outerBegin = nullptr) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitArrayInitLoopExpr");
  }
  void VisitImplicitValueInitExpr(ImplicitValueInitExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitImplicitValueInitExpr");
  }
  void VisitNoInitExpr(NoInitExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitNoInitExpr");
  }
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    cgf.cgm.errorNYI(dae->getSourceRange(),
                     "AggExprEmitter: VisitCXXDefaultArgExpr");
  }
  void VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    cgf.cgm.errorNYI(die->getSourceRange(),
                     "AggExprEmitter: VisitCXXDefaultInitExpr");
  }
  void VisitCXXInheritedCtorInitExpr(const CXXInheritedCtorInitExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitCXXInheritedCtorInitExpr");
  }
  void VisitLambdaExpr(LambdaExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitLambdaExpr");
  }
  void VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitCXXStdInitializerListExpr");
  }

  void VisitExprWithCleanups(ExprWithCleanups *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitExprWithCleanups");
  }
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitCXXScalarValueInitExpr");
  }
  void VisitCXXTypeidExpr(CXXTypeidExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitCXXTypeidExpr");
  }
  void VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitMaterializeTemporaryExpr");
  }
  void VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitOpaqueValueExpr");
  }

  void VisitPseudoObjectExpr(PseudoObjectExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "AggExprEmitter: VisitPseudoObjectExpr");
  }

  void VisitVAArgExpr(VAArgExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitVAArgExpr");
  }

  void VisitCXXThrowExpr(const CXXThrowExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitCXXThrowExpr");
  }
  void VisitAtomicExpr(AtomicExpr *e) {
    cgf.cgm.errorNYI(e->getSourceRange(), "AggExprEmitter: VisitAtomicExpr");
  }
};

} // namespace

static bool isTrivialFiller(Expr *e) {
  if (!e)
    return true;

  if (isa<ImplicitValueInitExpr>(e))
    return true;

  if (auto *ile = dyn_cast<InitListExpr>(e)) {
    if (ile->getNumInits())
      return false;
    return isTrivialFiller(ile->getArrayFiller());
  }

  if (const auto *cons = dyn_cast_or_null<CXXConstructExpr>(e))
    return cons->getConstructor()->isDefaultConstructor() &&
           cons->getConstructor()->isTrivial();

  return false;
}

/// Given an expression with aggregate type that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result into DestPtr.
void AggExprEmitter::emitAggLoadOfLValue(const Expr *e) {
  LValue lv = cgf.emitLValue(e);

  // If the type of the l-value is atomic, then do an atomic load.
  assert(!cir::MissingFeatures::opLoadStoreAtomic());

  emitFinalDestCopy(e->getType(), lv);
}

void AggExprEmitter::emitArrayInit(Address destPtr, cir::ArrayType arrayTy,
                                   QualType arrayQTy, Expr *e,
                                   ArrayRef<Expr *> args, Expr *arrayFiller) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  const mlir::Location loc = cgf.getLoc(e->getSourceRange());

  const uint64_t numInitElements = args.size();

  const QualType elementType =
      cgf.getContext().getAsArrayType(arrayQTy)->getElementType();

  if (elementType.isDestructedType() && cgf.cgm.getLangOpts().Exceptions) {
    cgf.cgm.errorNYI(loc, "initialized array requires destruction");
    return;
  }

  const QualType elementPtrType = cgf.getContext().getPointerType(elementType);

  const mlir::Type cirElementType = cgf.convertType(elementType);
  const cir::PointerType cirElementPtrType =
      builder.getPointerTo(cirElementType);

  auto begin = cir::CastOp::create(builder, loc, cirElementPtrType,
                                   cir::CastKind::array_to_ptrdecay,
                                   destPtr.getPointer());

  const CharUnits elementSize =
      cgf.getContext().getTypeSizeInChars(elementType);
  const CharUnits elementAlign =
      destPtr.getAlignment().alignmentOfArrayElement(elementSize);

  // The 'current element to initialize'.  The invariants on this
  // variable are complicated.  Essentially, after each iteration of
  // the loop, it points to the last initialized element, except
  // that it points to the beginning of the array before any
  // elements have been initialized.
  mlir::Value element = begin;

  // Don't build the 'one' before the cycle to avoid
  // emmiting the redundant `cir.const 1` instrs.
  mlir::Value one;

  // Emit the explicit initializers.
  for (uint64_t i = 0; i != numInitElements; ++i) {
    // Advance to the next element.
    if (i > 0) {
      one = builder.getConstantInt(loc, cgf.PtrDiffTy, i);
      element = builder.createPtrStride(loc, begin, one);
    }

    const Address address = Address(element, cirElementType, elementAlign);
    const LValue elementLV = cgf.makeAddrLValue(address, elementType);
    emitInitializationToLValue(args[i], elementLV);
  }

  const uint64_t numArrayElements = arrayTy.getSize();

  // Check whether there's a non-trivial array-fill expression.
  const bool hasTrivialFiller = isTrivialFiller(arrayFiller);

  // Any remaining elements need to be zero-initialized, possibly
  // using the filler expression.  We can skip this if the we're
  // emitting to zeroed memory.
  if (numInitElements != numArrayElements &&
      !(dest.isZeroed() && hasTrivialFiller &&
        cgf.getTypes().isZeroInitializable(elementType))) {
    // Advance to the start of the rest of the array.
    if (numInitElements) {
      one = builder.getConstantInt(loc, cgf.PtrDiffTy, 1);
      element = cir::PtrStrideOp::create(builder, loc, cirElementPtrType,
                                         element, one);
    }

    // Allocate the temporary variable
    // to store the pointer to first unitialized element
    const Address tmpAddr = cgf.createTempAlloca(
        cirElementPtrType, cgf.getPointerAlign(), loc, "arrayinit.temp");
    LValue tmpLV = cgf.makeAddrLValue(tmpAddr, elementPtrType);
    cgf.emitStoreThroughLValue(RValue::get(element), tmpLV);

    // Compute the end of array
    cir::ConstantOp numArrayElementsConst = builder.getConstInt(
        loc, mlir::cast<cir::IntType>(cgf.PtrDiffTy), numArrayElements);
    mlir::Value end = cir::PtrStrideOp::create(builder, loc, cirElementPtrType,
                                               begin, numArrayElementsConst);

    builder.createDoWhile(
        loc,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          cir::LoadOp currentElement = builder.createLoad(loc, tmpAddr);
          mlir::Type boolTy = cgf.convertType(cgf.getContext().BoolTy);
          cir::CmpOp cmp = cir::CmpOp::create(
              builder, loc, boolTy, cir::CmpOpKind::ne, currentElement, end);
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          cir::LoadOp currentElement = builder.createLoad(loc, tmpAddr);

          assert(!cir::MissingFeatures::requiresCleanups());

          // Emit the actual filler expression.
          LValue elementLV = cgf.makeAddrLValue(
              Address(currentElement, cirElementType, elementAlign),
              elementType);
          if (arrayFiller)
            emitInitializationToLValue(arrayFiller, elementLV);
          else
            emitNullInitializationToLValue(loc, elementLV);

          // Tell the EH cleanup that we finished with the last element.
          if (cgf.cgm.getLangOpts().Exceptions) {
            cgf.cgm.errorNYI(loc, "update destructed array element for EH");
            return;
          }

          // Advance pointer and store them to temporary variable
          cir::ConstantOp one = builder.getConstInt(
              loc, mlir::cast<cir::IntType>(cgf.PtrDiffTy), 1);
          auto nextElement = cir::PtrStrideOp::create(
              builder, loc, cirElementPtrType, currentElement, one);
          cgf.emitStoreThroughLValue(RValue::get(nextElement), tmpLV);

          builder.createYield(loc);
        });
  }
}

/// Perform the final copy to destPtr, if desired.
void AggExprEmitter::emitFinalDestCopy(QualType type, const LValue &src) {
  // If dest is ignored, then we're evaluating an aggregate expression
  // in a context that doesn't care about the result.  Note that loads
  // from volatile l-values force the existence of a non-ignored
  // destination.
  if (dest.isIgnored())
    return;

  cgf.cgm.errorNYI("emitFinalDestCopy: non-ignored dest is NYI");
}

void AggExprEmitter::emitInitializationToLValue(Expr *e, LValue lv) {
  const QualType type = lv.getType();

  if (isa<ImplicitValueInitExpr, CXXScalarValueInitExpr>(e)) {
    const mlir::Location loc = e->getSourceRange().isValid()
                                   ? cgf.getLoc(e->getSourceRange())
                                   : *cgf.currSrcLoc;
    return emitNullInitializationToLValue(loc, lv);
  }

  if (isa<NoInitExpr>(e))
    return;

  if (type->isReferenceType())
    cgf.cgm.errorNYI("emitInitializationToLValue ReferenceType");

  switch (cgf.getEvaluationKind(type)) {
  case cir::TEK_Complex:
    cgf.cgm.errorNYI("emitInitializationToLValue TEK_Complex");
    break;
  case cir::TEK_Aggregate:
    cgf.emitAggExpr(e, AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                               AggValueSlot::IsNotAliased,
                                               AggValueSlot::MayOverlap,
                                               dest.isZeroed()));

    return;
  case cir::TEK_Scalar:
    if (lv.isSimple())
      cgf.emitScalarInit(e, cgf.getLoc(e->getSourceRange()), lv);
    else
      cgf.emitStoreThroughLValue(RValue::get(cgf.emitScalarExpr(e)), lv);
    return;
  }
}

void AggExprEmitter::VisitCXXConstructExpr(const CXXConstructExpr *e) {
  AggValueSlot slot = ensureSlot(cgf.getLoc(e->getSourceRange()), e->getType());
  cgf.emitCXXConstructExpr(e, slot);
}

void AggExprEmitter::emitNullInitializationToLValue(mlir::Location loc,
                                                    LValue lv) {
  const QualType type = lv.getType();

  // If the destination slot is already zeroed out before the aggregate is
  // copied into it, we don't have to emit any zeros here.
  if (dest.isZeroed() && cgf.getTypes().isZeroInitializable(type))
    return;

  if (cgf.hasScalarEvaluationKind(type)) {
    // For non-aggregates, we can store the appropriate null constant.
    mlir::Value null = cgf.cgm.emitNullConstant(type, loc);
    if (lv.isSimple()) {
      cgf.emitStoreOfScalar(null, lv, /* isInitialization */ true);
      return;
    }

    cgf.cgm.errorNYI("emitStoreThroughBitfieldLValue");
    return;
  }

  // There's a potential optimization opportunity in combining
  // memsets; that would be easy for arrays, but relatively
  // difficult for structures with the current code.
  cgf.emitNullInitialization(loc, lv.getAddress(), lv.getType());
}

void AggExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType()) {
    cgf.cgm.errorNYI(e->getSourceRange(), "reference return type");
    return;
  }

  withReturnValueSlot(
      e, [&](ReturnValueSlot slot) { return cgf.emitCallExpr(e, slot); });
}

void AggExprEmitter::withReturnValueSlot(
    const Expr *e, llvm::function_ref<RValue(ReturnValueSlot)> fn) {
  QualType retTy = e->getType();

  assert(!cir::MissingFeatures::aggValueSlotDestructedFlag());
  bool requiresDestruction =
      retTy.isDestructedType() == QualType::DK_nontrivial_c_struct;
  if (requiresDestruction)
    cgf.cgm.errorNYI(
        e->getSourceRange(),
        "withReturnValueSlot: return value requiring destruction is NYI");

  // If it makes no observable difference, save a memcpy + temporary.
  //
  // We need to always provide our own temporary if destruction is required.
  // Otherwise, fn will emit its own, notice that it's "unused", and end its
  // lifetime before we have the chance to emit a proper destructor call.
  assert(!cir::MissingFeatures::aggValueSlotAlias());
  assert(!cir::MissingFeatures::aggValueSlotGC());

  Address retAddr = dest.getAddress();
  assert(!cir::MissingFeatures::emitLifetimeMarkers());

  assert(!cir::MissingFeatures::aggValueSlotVolatile());
  assert(!cir::MissingFeatures::aggValueSlotDestructedFlag());
  fn(ReturnValueSlot(retAddr));
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *e) {
  if (e->hadArrayRangeDesignator())
    llvm_unreachable("GNU array range designator extension");

  if (e->isTransparent())
    return Visit(e->getInit(0));

  visitCXXParenListOrInitListExpr(
      e, e->inits(), e->getInitializedFieldInUnion(), e->getArrayFiller());
}

void AggExprEmitter::visitCXXParenListOrInitListExpr(
    Expr *e, ArrayRef<Expr *> args, FieldDecl *initializedFieldInUnion,
    Expr *arrayFiller) {

  const AggValueSlot dest =
      ensureSlot(cgf.getLoc(e->getSourceRange()), e->getType());

  if (e->getType()->isConstantArrayType()) {
    cir::ArrayType arrayTy =
        cast<cir::ArrayType>(dest.getAddress().getElementType());
    emitArrayInit(dest.getAddress(), arrayTy, e->getType(), e, args,
                  arrayFiller);
    return;
  } else if (e->getType()->isVariableArrayType()) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "visitCXXParenListOrInitListExpr variable array type");
    return;
  }

  if (e->getType()->isArrayType()) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "visitCXXParenListOrInitListExpr array type");
    return;
  }

  assert(e->getType()->isRecordType() && "Only support structs/unions here!");

  // Do struct initialization; this code just sets each individual member
  // to the approprate value.  This makes bitfield support automatic;
  // the disadvantage is that the generated code is more difficult for
  // the optimizer, especially with bitfields.
  unsigned numInitElements = args.size();
  auto *record = e->getType()->castAsRecordDecl();

  // We'll need to enter cleanup scopes in case any of the element
  // initializers throws an exception.
  assert(!cir::MissingFeatures::requiresCleanups());

  unsigned curInitIndex = 0;

  // Emit initialization of base classes.
  if (auto *cxxrd = dyn_cast<CXXRecordDecl>(record)) {
    assert(numInitElements >= cxxrd->getNumBases() &&
           "missing initializer for base class");
    if (cxxrd->getNumBases() > 0) {
      cgf.cgm.errorNYI(e->getSourceRange(),
                       "visitCXXParenListOrInitListExpr base class init");
      return;
    }
  }

  LValue destLV = cgf.makeAddrLValue(dest.getAddress(), e->getType());

  if (record->isUnion()) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "visitCXXParenListOrInitListExpr union type");
    return;
  }

  // Here we iterate over the fields; this makes it simpler to both
  // default-initialize fields and skip over unnamed fields.
  for (const FieldDecl *field : record->fields()) {
    // We're done once we hit the flexible array member.
    if (field->getType()->isIncompleteArrayType())
      break;

    // Always skip anonymous bitfields.
    if (field->isUnnamedBitField())
      continue;

    // We're done if we reach the end of the explicit initializers, we
    // have a zeroed object, and the rest of the fields are
    // zero-initializable.
    if (curInitIndex == numInitElements && dest.isZeroed() &&
        cgf.getTypes().isZeroInitializable(e->getType()))
      break;
    LValue lv =
        cgf.emitLValueForFieldInitialization(destLV, field, field->getName());
    // We never generate write-barriers for initialized fields.
    assert(!cir::MissingFeatures::setNonGC());

    if (curInitIndex < numInitElements) {
      // Store the initializer into the field.
      CIRGenFunction::SourceLocRAIIObject loc{
          cgf, cgf.getLoc(record->getSourceRange())};
      emitInitializationToLValue(args[curInitIndex++], lv);
    } else {
      // We're out of initializers; default-initialize to null
      emitNullInitializationToLValue(cgf.getLoc(e->getSourceRange()), lv);
    }

    // Push a destructor if necessary.
    // FIXME: if we have an array of structures, all explicitly
    // initialized, we can end up pushing a linear number of cleanups.
    if (field->getType().isDestructedType()) {
      cgf.cgm.errorNYI(e->getSourceRange(),
                       "visitCXXParenListOrInitListExpr destructor");
      return;
    }

    // From classic codegen, maybe not useful for CIR:
    // If the GEP didn't get used because of a dead zero init or something
    // else, clean it up for -O0 builds and general tidiness.
  }
}

// TODO(cir): This could be shared with classic codegen.
AggValueSlot::Overlap_t CIRGenFunction::getOverlapForBaseInit(
    const CXXRecordDecl *rd, const CXXRecordDecl *baseRD, bool isVirtual) {
  // If the most-derived object is a field declared with [[no_unique_address]],
  // the tail padding of any virtual base could be reused for other subobjects
  // of that field's class.
  if (isVirtual)
    return AggValueSlot::MayOverlap;

  // If the base class is laid out entirely within the nvsize of the derived
  // class, its tail padding cannot yet be initialized, so we can issue
  // stores at the full width of the base class.
  const ASTRecordLayout &layout = getContext().getASTRecordLayout(rd);
  if (layout.getBaseClassOffset(baseRD) +
          getContext().getASTRecordLayout(baseRD).getSize() <=
      layout.getNonVirtualSize())
    return AggValueSlot::DoesNotOverlap;

  // The tail padding may contain values we need to preserve.
  return AggValueSlot::MayOverlap;
}

void CIRGenFunction::emitAggExpr(const Expr *e, AggValueSlot slot) {
  AggExprEmitter(*this, slot).Visit(const_cast<Expr *>(e));
}

LValue CIRGenFunction::emitAggExprToLValue(const Expr *e) {
  assert(hasAggregateEvaluationKind(e->getType()) && "Invalid argument!");
  Address temp = createMemTemp(e->getType(), getLoc(e->getSourceRange()));
  LValue lv = makeAddrLValue(temp, e->getType());
  emitAggExpr(e, AggValueSlot::forLValue(lv, AggValueSlot::IsNotDestructed,
                                         AggValueSlot::IsNotAliased,
                                         AggValueSlot::DoesNotOverlap));
  return lv;
}
