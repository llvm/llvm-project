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

#include "clang/AST/StmtVisitor.h"

using namespace cir;
using namespace clang;

namespace {
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

  AggValueSlot EnsureSlot(QualType T) {
    assert(!Dest.isIgnored() && "ignored slots NYI");
    return Dest;
  }

public:
  AggExprEmitter(CIRGenFunction &cgf, AggValueSlot Dest, bool IsResultUnused)
      : CGF{cgf}, Dest(Dest), IsResultUnused(IsResultUnused) {}

  //===--------------------------------------------------------------------===//
  //                             Visitor Methods
  //===--------------------------------------------------------------------===//

  void Visit(Expr *E) {
    if (CGF.getDebugInfo()) {
      llvm_unreachable("NYI");
    }
    StmtVisitor<AggExprEmitter>::Visit(E);
  }

  void VisitStmt(Stmt *S) { llvm_unreachable("NYI"); }
  void VisitParenExpr(ParenExpr *PE) { llvm_unreachable("NYI"); }
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
  void VisitDeclRefExpr(DeclRefExpr *E) { llvm_unreachable("NYI"); }
  void VisitMemberExpr(MemberExpr *E) { llvm_unreachable("NYI"); }
  void VisitUnaryDeref(UnaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitStringLiteral(StringLiteral *E) { llvm_unreachable("NYI"); }
  void VisitCompoundLIteralExpr(CompoundLiteralExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitPredefinedExpr(const PredefinedExpr *E) { llvm_unreachable("NYI"); }

  // Operators.
  void VisitCastExpr(CastExpr *E);
  void VisitCallExpr(const CallExpr *E);
  void VisitStmtExpr(const StmtExpr *E) { llvm_unreachable("NYI"); }
  void VisitBinaryOperator(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitPointerToDataMemberBinaryOperator(const BinaryOperator *E) {
    llvm_unreachable("NYI");
  }
  void VisitBinAssign(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitBinComma(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitBinCmp(const BinaryOperator *E) { llvm_unreachable("NYI"); }
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
  void VisitArrayInitLoopExpr(const ArrayInitLoopExpr *E,
                              llvm::Value *outerBegin = nullptr) {
    llvm_unreachable("NYI");
  }
  void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitNoInitExpr(NoInitExpr *E) { llvm_unreachable("NYI"); }
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) { llvm_unreachable("NYI"); }
  void VisitXCXDefaultInitExpr(CXXDefaultInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
    llvm_unreachable("NYI");
  }
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

  void EmitInitializationToLValue(Expr *E, LValue LV);

  void EmitNullInitializationToLValue(LValue Address) {
    llvm_unreachable("NYI");
  }
  // case Expr::ChoseExprClass:
  void VisitCXXThrowExpr(const CXXThrowExpr *E) { llvm_unreachable("NYI"); }
  void VisitAtomicExpr(AtomicExpr *E) { llvm_unreachable("NYI"); }
};
} // namespace

//===----------------------------------------------------------------------===//
//                             Visitor Methods
//===----------------------------------------------------------------------===//

/// If emitting this value will obviously just cause a store of
/// zero to memory, return true.  This can return false if uncertain, so it just
/// handles simple cases.
static bool isSimpleZero(const Expr *E, CIRGenFunction &CGF) {
  E = E->IgnoreParens();
  while (auto *CE = dyn_cast<CastExpr>(E)) {
    llvm_unreachable("NYI");
    // if (!castPreservesZero(CE))
    //   break;
    // E = CE->getSubExpr()->IgnoreParens();
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
    llvm_unreachable("NYI");
    // return ICE->getCastKind() == CK_NullToPointer &&
    //        CGF.getTypes().isPointerZeroInitializable(E->getType()) &&
    //        !E->HasSideEffects(CGF.getContext());
  }
  // '\0'
  if (const CharacterLiteral *CL = dyn_cast<CharacterLiteral>(E))
    return CL->getValue() == 0;

  // Otherwise, hard case: conservatively return false.
  return false;
}

void AggExprEmitter::EmitInitializationToLValue(Expr *E, LValue LV) {
  QualType type = LV.getType();
  // FIXME: Ignore result?
  // FIXME: Are initializers affected by volatile?
  if (Dest.isZeroed() && isSimpleZero(E, CGF)) {
    // TODO(cir): LLVM codegen just returns here, do we want to
    // do anything different when we hit this code path?
    llvm_unreachable("NYI");
    // Storing "i32 0" to a zero'd memory location is a noop.
    return;
  } else if (isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E)) {
    return EmitNullInitializationToLValue(LV);
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
      llvm_unreachable("NYI");
      // CGF.EmitStoreThroughLValue(RValue::get(CGF.EmitScalarExpr(E)), LV);
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
  AggValueSlot Slot = EnsureSlot(E->getType());
  CGF.buildCXXConstructExpr(E, Slot);
}

void AggExprEmitter::VisitExprWithCleanups(ExprWithCleanups *E) {
  if (UnimplementedFeature::cleanups())
    llvm_unreachable("NYI");

  auto &builder = CGF.getBuilder();
  auto scopeLoc = CGF.getLoc(E->getSourceRange());
  [[maybe_unused]] auto scope = builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        SmallVector<mlir::Location, 2> locs;
        if (loc.isa<mlir::FileLineColLoc>()) {
          locs.push_back(loc);
          locs.push_back(loc);
        } else if (loc.isa<mlir::FusedLoc>()) {
          auto fusedLoc = loc.cast<mlir::FusedLoc>();
          locs.push_back(fusedLoc.getLocations()[0]);
          locs.push_back(fusedLoc.getLocations()[1]);
        }
        CIRGenFunction::LexicalScopeContext lexScope{
            locs[0], locs[1], builder.getInsertionBlock()};
        CIRGenFunction::LexicalScopeGuard lexScopeGuard{CGF, &lexScope};
        Visit(E->getSubExpr());
      });
}

void AggExprEmitter::VisitLambdaExpr(LambdaExpr *E) {
  CIRGenFunction::SourceLocRAIIObject loc{CGF, CGF.getLoc(E->getSourceRange())};
  AggValueSlot Slot = EnsureSlot(E->getType());
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

    EmitInitializationToLValue(captureInit, LV);

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
    llvm_unreachable("NYI");
  }

  RValue Src =
      EmitCall(ReturnValueSlot(RetAddr, Dest.isVolatile(), IsResultUnused,
                               Dest.isExternallyDestructed()));

  if (!UseTemp)
    return;

  assert(Dest.isIgnored() || Dest.getPointer() != Src.getAggregatePointer());
  llvm_unreachable("NYI");
  // TODO(cir): EmitFinalDestCopy(E->getType(), Src);

  if (!RequiresDestruction) {
    // If there's no dtor to run, the copy was the last use of our temporary.
    // Since we're not guaranteed to be in an ExprWithCleanups, clean up
    // eagerly.
    llvm_unreachable("NYI");
  }
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *E) {
  // If the initializer list is empty ({}), and there are
  // no explicitly initialized elements.
  if (E->getNumInits() == 0)
    return;

  // TODO(cir): use something like CGF.ErrorUnsupported
  if (E->hadArrayRangeDesignator())
    llvm_unreachable("GNU array range designator extension");

  if (E->isTransparent())
    return Visit(E->getInit(0));

  AggValueSlot Dest = EnsureSlot(E->getType());

  [[maybe_unused]] LValue DestLV =
      CGF.makeAddrLValue(Dest.getAddress(), E->getType());

  // Handle initialization of an array.
  if (E->getType()->isArrayType()) {
    llvm_unreachable("NYI");
  }

  assert(E->getType()->isRecordType() && "Only support structs/unions here!");
  llvm_unreachable("NYI");
}

//===----------------------------------------------------------------------===//
//                        Helpers and dispatcher
//===----------------------------------------------------------------------===//

/// CheckAggExprForMemSetUse - If the initializer is large and has a lot of
/// zeros in it, emit a memset and avoid storing the individual zeros.
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

  llvm_unreachable("NYI");
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

  // Address DestPtr = Dest.getAddress();
  // Address SrcPtr = Src.getAddress();

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
    assert(0 && "NYI");
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

  llvm::Value *SizeVal = nullptr;
  if (TypeInfo.Width.isZero()) {
    assert(0 && "NYI");
  }
  if (!SizeVal) {
    assert(0 && "NYI");
    // SizeVal = llvm::ConstantInt::get(SizeTy, TypeInfo.Width.getQuantity());
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

  assert(0 && "NYI");
  // DestPtr = Builder.CreateElementBitCast(DestPtr, Int8Ty);
  // SrcPtr = Builder.CreateElementBitCast(SrcPtr, Int8Ty);

  // Don't do any of the memmove_collectable tests if GC isn't set.
  if (CGM.getLangOpts().getGC() == LangOptions::NonGC) {
    // fall through
  } else if (const RecordType *RecordTy = Ty->getAs<RecordType>()) {
    assert(0 && "NYI");
  } else if (Ty->isArrayType()) {
    assert(0 && "NYI");
  }

  assert(0 && "NYI");
  // auto Inst = Builder.CreateMemCpy(DestPtr, SrcPtr, SizeVal, isVolatile);

  // Determine the metadata to describe the position of any padding in this
  // memcpy, as well as the TBAA tags for the members of the struct, in case
  // the optimizer wishes to expand it in to scalar memory operations.
  assert(!UnimplementedFeature::tbaa());
  if (CGM.getCodeGenOpts().NewStructPathTBAA) {
    assert(0 && "NYI");
  }
}
