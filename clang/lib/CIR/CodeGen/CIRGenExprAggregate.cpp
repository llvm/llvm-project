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

  void VisitDeclRefExpr(DeclRefExpr *e) { emitAggLoadOfLValue(e); }

  void VisitInitListExpr(InitListExpr *e);
  void VisitCXXConstructExpr(const CXXConstructExpr *e);

  void visitCXXParenListOrInitListExpr(Expr *e, ArrayRef<Expr *> args,
                                       FieldDecl *initializedFieldInUnion,
                                       Expr *arrayFiller);
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

  if (elementType.isDestructedType()) {
    cgf.cgm.errorNYI(loc, "dtorKind NYI");
    return;
  }

  const QualType elementPtrType = cgf.getContext().getPointerType(elementType);

  const mlir::Type cirElementType = cgf.convertType(elementType);
  const cir::PointerType cirElementPtrType =
      builder.getPointerTo(cirElementType);

  auto begin = builder.create<cir::CastOp>(loc, cirElementPtrType,
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
      element = builder.create<cir::PtrStrideOp>(loc, cirElementPtrType,
                                                 element, one);
    }

    // Allocate the temporary variable
    // to store the pointer to first unitialized element
    const Address tmpAddr = cgf.createTempAlloca(
        cirElementPtrType, cgf.getPointerAlign(), loc, "arrayinit.temp");
    LValue tmpLV = cgf.makeAddrLValue(tmpAddr, elementPtrType);
    cgf.emitStoreThroughLValue(RValue::get(element), tmpLV);

    // TODO(CIR): Replace this part later with cir::DoWhileOp
    for (unsigned i = numInitElements; i != numArrayElements; ++i) {
      cir::LoadOp currentElement = builder.createLoad(loc, tmpAddr);

      // Emit the actual filler expression.
      const LValue elementLV = cgf.makeAddrLValue(
          Address(currentElement, cirElementType, elementAlign), elementType);

      if (arrayFiller)
        emitInitializationToLValue(arrayFiller, elementLV);
      else
        emitNullInitializationToLValue(loc, elementLV);

      // Advance pointer and store them to temporary variable
      one = builder.getConstantInt(loc, cgf.PtrDiffTy, 1);
      cir::PtrStrideOp nextElement =
          builder.createPtrStride(loc, currentElement, one);
      cgf.emitStoreThroughLValue(RValue::get(nextElement), tmpLV);
    }
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
  }

  cgf.cgm.errorNYI(
      "visitCXXParenListOrInitListExpr Record or VariableSizeArray type");
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
