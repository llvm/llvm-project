//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Constant Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CIRGenConstantEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenRecordLayout.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Specifiers.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//

// This class only needs to handle arrays, structs and unions.
//
// In LLVM codegen, when outside C++11 mode, those types are not constant
// folded, while all other types are handled by constant folding.
//
// In CIR codegen, instead of folding things here, we should defer that work
// to MLIR: do not attempt to do much here.
class ConstExprEmitter
    : public StmtVisitor<ConstExprEmitter, mlir::Attribute, QualType> {
  CIRGenModule &cgm;
  LLVM_ATTRIBUTE_UNUSED ConstantEmitter &emitter;

public:
  ConstExprEmitter(ConstantEmitter &emitter)
      : cgm(emitter.cgm), emitter(emitter) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Attribute VisitStmt(Stmt *S, QualType T) { return {}; }

  mlir::Attribute VisitConstantExpr(ConstantExpr *ce, QualType t) {
    if (mlir::Attribute result = emitter.tryEmitConstantExpr(ce))
      return result;
    return Visit(ce->getSubExpr(), t);
  }

  mlir::Attribute VisitParenExpr(ParenExpr *pe, QualType t) {
    return Visit(pe->getSubExpr(), t);
  }

  mlir::Attribute
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *pe,
                                    QualType t) {
    return Visit(pe->getReplacement(), t);
  }

  mlir::Attribute VisitGenericSelectionExpr(GenericSelectionExpr *ge,
                                            QualType t) {
    return Visit(ge->getResultExpr(), t);
  }

  mlir::Attribute VisitChooseExpr(ChooseExpr *ce, QualType t) {
    return Visit(ce->getChosenSubExpr(), t);
  }

  mlir::Attribute VisitCompoundLiteralExpr(CompoundLiteralExpr *e, QualType t) {
    return Visit(e->getInitializer(), t);
  }

  mlir::Attribute VisitCastExpr(CastExpr *e, QualType destType) {
    if (isa<ExplicitCastExpr>(e))
      cgm.errorNYI(e->getBeginLoc(),
                   "ConstExprEmitter::VisitCastExpr explicit cast");

    Expr *subExpr = e->getSubExpr();

    switch (e->getCastKind()) {
    case CK_ToUnion:
    case CK_AddressSpaceConversion:
    case CK_ReinterpretMemberPointer:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer:
      cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitCastExpr");
      return {};

    case CK_LValueToRValue:
    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_ConstructorConversion:
      return Visit(subExpr, destType);

    case CK_IntToOCLSampler:
      llvm_unreachable("global sampler variables are not generated");

    case CK_Dependent:
      llvm_unreachable("saw dependent cast!");

    case CK_BuiltinFnToFnPtr:
      llvm_unreachable("builtin functions are handled elsewhere");

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_ARCProduceObject:
    case CK_ARCConsumeObject:
    case CK_ARCReclaimReturnedObject:
    case CK_ARCExtendBlockObject:
    case CK_CopyAndAutoreleaseBlockObject:
      return {};

    // These don't need to be handled here because Evaluate knows how to
    // evaluate them in the cases where they can be folded.
    case CK_BitCast:
    case CK_ToVoid:
    case CK_Dynamic:
    case CK_LValueBitCast:
    case CK_LValueToRValueBitCast:
    case CK_NullToMemberPointer:
    case CK_UserDefinedConversion:
    case CK_CPointerToObjCPointerCast:
    case CK_BlockPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_MemberPointerToBoolean:
    case CK_VectorSplat:
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
    case CK_PointerToIntegral:
    case CK_PointerToBoolean:
    case CK_NullToPointer:
    case CK_IntegralCast:
    case CK_BooleanToSignedIntegral:
    case CK_IntegralToPointer:
    case CK_IntegralToBoolean:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingToBoolean:
    case CK_FloatingCast:
    case CK_FloatingToFixedPoint:
    case CK_FixedPointToFloating:
    case CK_FixedPointCast:
    case CK_FixedPointToBoolean:
    case CK_FixedPointToIntegral:
    case CK_IntegralToFixedPoint:
    case CK_ZeroToOCLOpaqueType:
    case CK_MatrixCast:
    case CK_HLSLArrayRValue:
    case CK_HLSLVectorTruncation:
    case CK_HLSLElementwiseCast:
    case CK_HLSLAggregateSplatCast:
      return {};
    }
    llvm_unreachable("Invalid CastKind");
  }

  mlir::Attribute VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die, QualType t) {
    cgm.errorNYI(die->getBeginLoc(),
                 "ConstExprEmitter::VisitCXXDefaultInitExpr");
    return {};
  }

  mlir::Attribute VisitExprWithCleanups(ExprWithCleanups *e, QualType t) {
    // Since this about constant emission no need to wrap this under a scope.
    return Visit(e->getSubExpr(), t);
  }

  mlir::Attribute VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *e,
                                                QualType t) {
    return Visit(e->getSubExpr(), t);
  }

  mlir::Attribute VisitImplicitValueInitExpr(ImplicitValueInitExpr *E,
                                             QualType T) {
    cgm.errorNYI(E->getBeginLoc(),
                 "ConstExprEmitter::VisitImplicitValueInitExpr");
    return {};
  }

  mlir::Attribute VisitInitListExpr(InitListExpr *ile, QualType t) {
    if (ile->isTransparent())
      return Visit(ile->getInit(0), t);

    if (ile->getType()->isArrayType()) {
      // If we return null here, the non-constant initializer will take care of
      // it, but we would prefer to handle it here.
      assert(!cir::MissingFeatures::constEmitterArrayILE());
      return {};
    }

    if (ile->getType()->isRecordType()) {
      cgm.errorNYI(ile->getBeginLoc(), "ConstExprEmitter: record ILE");
      return {};
    }

    if (ile->getType()->isVectorType()) {
      // If we return null here, the non-constant initializer will take care of
      // it, but we would prefer to handle it here.
      assert(!cir::MissingFeatures::constEmitterVectorILE());
      return {};
    }

    return {};
  }

  mlir::Attribute VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *e,
                                                QualType destType) {
    mlir::Attribute c = Visit(e->getBase(), destType);
    if (!c)
      return {};

    cgm.errorNYI(e->getBeginLoc(),
                 "ConstExprEmitter::VisitDesignatedInitUpdateExpr");
    return {};
  }

  mlir::Attribute VisitCXXConstructExpr(CXXConstructExpr *e, QualType ty) {
    cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitCXXConstructExpr");
    return {};
  }

  mlir::Attribute VisitStringLiteral(StringLiteral *e, QualType t) {
    // This is a string literal initializing an array in an initializer.
    return cgm.getConstantArrayFromStringLiteral(e);
  }

  mlir::Attribute VisitObjCEncodeExpr(ObjCEncodeExpr *e, QualType t) {
    cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitObjCEncodeExpr");
    return {};
  }

  mlir::Attribute VisitUnaryExtension(const UnaryOperator *e, QualType t) {
    return Visit(e->getSubExpr(), t);
  }

  // Utility methods
  mlir::Type convertType(QualType t) { return cgm.convertType(t); }
};

// TODO(cir): this can be shared with LLVM's codegen
static QualType getNonMemoryType(CIRGenModule &cgm, QualType type) {
  if (const auto *at = type->getAs<AtomicType>()) {
    return cgm.getASTContext().getQualifiedType(at->getValueType(),
                                                type.getQualifiers());
  }
  return type;
}

static mlir::Attribute
emitArrayConstant(CIRGenModule &cgm, mlir::Type desiredType,
                  mlir::Type commonElementType, unsigned arrayBound,
                  SmallVectorImpl<mlir::TypedAttr> &elements,
                  mlir::TypedAttr filler) {
  const CIRGenBuilderTy &builder = cgm.getBuilder();

  unsigned nonzeroLength = arrayBound;
  if (elements.size() < nonzeroLength && builder.isNullValue(filler))
    nonzeroLength = elements.size();

  if (nonzeroLength == elements.size()) {
    while (nonzeroLength > 0 &&
           builder.isNullValue(elements[nonzeroLength - 1]))
      --nonzeroLength;
  }

  if (nonzeroLength == 0)
    return cir::ZeroAttr::get(desiredType);

  const unsigned trailingZeroes = arrayBound - nonzeroLength;

  // Add a zeroinitializer array filler if we have lots of trailing zeroes.
  if (trailingZeroes >= 8) {
    assert(elements.size() >= nonzeroLength &&
           "missing initializer for non-zero element");
  } else if (elements.size() != arrayBound) {
    elements.resize(arrayBound, filler);

    if (filler.getType() != commonElementType)
      commonElementType = {};
  }

  if (commonElementType) {
    SmallVector<mlir::Attribute, 4> eles;
    eles.reserve(elements.size());

    for (const auto &element : elements)
      eles.push_back(element);

    return cir::ConstArrayAttr::get(
        cir::ArrayType::get(commonElementType, arrayBound),
        mlir::ArrayAttr::get(builder.getContext(), eles));
  }

  cgm.errorNYI("array with different type elements");
  return {};
}

//===----------------------------------------------------------------------===//
//                          ConstantLValueEmitter
//===----------------------------------------------------------------------===//

namespace {
/// A struct which can be used to peephole certain kinds of finalization
/// that normally happen during l-value emission.
struct ConstantLValue {
  llvm::PointerUnion<mlir::Value, mlir::Attribute> value;
  bool hasOffsetApplied;

  ConstantLValue(std::nullptr_t) : value(nullptr), hasOffsetApplied(false) {}
  ConstantLValue() : value(nullptr), hasOffsetApplied(false) {}
};

/// A helper class for emitting constant l-values.
class ConstantLValueEmitter
    : public ConstStmtVisitor<ConstantLValueEmitter, ConstantLValue> {
  CIRGenModule &cgm;
  ConstantEmitter &emitter;
  const APValue &value;
  QualType destType;

  // Befriend StmtVisitorBase so that we don't have to expose Visit*.
  friend StmtVisitorBase;

public:
  ConstantLValueEmitter(ConstantEmitter &emitter, const APValue &value,
                        QualType destType)
      : cgm(emitter.cgm), emitter(emitter), value(value), destType(destType) {}

  mlir::Attribute tryEmit();

private:
  mlir::Attribute tryEmitAbsolute(mlir::Type destTy);
  ConstantLValue tryEmitBase(const APValue::LValueBase &base);

  ConstantLValue VisitStmt(const Stmt *s) { return nullptr; }
  ConstantLValue VisitConstantExpr(const ConstantExpr *e);
  ConstantLValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *e);
  ConstantLValue VisitStringLiteral(const StringLiteral *e);
  ConstantLValue VisitObjCBoxedExpr(const ObjCBoxedExpr *e);
  ConstantLValue VisitObjCEncodeExpr(const ObjCEncodeExpr *e);
  ConstantLValue VisitObjCStringLiteral(const ObjCStringLiteral *e);
  ConstantLValue VisitPredefinedExpr(const PredefinedExpr *e);
  ConstantLValue VisitAddrLabelExpr(const AddrLabelExpr *e);
  ConstantLValue VisitCallExpr(const CallExpr *e);
  ConstantLValue VisitBlockExpr(const BlockExpr *e);
  ConstantLValue VisitCXXTypeidExpr(const CXXTypeidExpr *e);
  ConstantLValue
  VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *e);
};

} // namespace

mlir::Attribute ConstantLValueEmitter::tryEmit() {
  const APValue::LValueBase &base = value.getLValueBase();

  // The destination type should be a pointer or reference
  // type, but it might also be a cast thereof.
  //
  // FIXME: the chain of casts required should be reflected in the APValue.
  // We need this in order to correctly handle things like a ptrtoint of a
  // non-zero null pointer and addrspace casts that aren't trivially
  // represented in LLVM IR.
  mlir::Type destTy = cgm.getTypes().convertTypeForMem(destType);
  assert(mlir::isa<cir::PointerType>(destTy));

  // If there's no base at all, this is a null or absolute pointer,
  // possibly cast back to an integer type.
  if (!base)
    return tryEmitAbsolute(destTy);

  // Otherwise, try to emit the base.
  ConstantLValue result = tryEmitBase(base);

  // If that failed, we're done.
  llvm::PointerUnion<mlir::Value, mlir::Attribute> &value = result.value;
  if (!value)
    return {};

  // Apply the offset if necessary and not already done.
  if (!result.hasOffsetApplied) {
    cgm.errorNYI("ConstantLValueEmitter: apply offset");
    return {};
  }

  // Convert to the appropriate type; this could be an lvalue for
  // an integer. FIXME: performAddrSpaceCast
  if (mlir::isa<cir::PointerType>(destTy)) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(value))
      return attr;
    cgm.errorNYI("ConstantLValueEmitter: non-attribute pointer");
    return {};
  }

  cgm.errorNYI("ConstantLValueEmitter: other?");
  return {};
}

/// Try to emit an absolute l-value, such as a null pointer or an integer
/// bitcast to pointer type.
mlir::Attribute ConstantLValueEmitter::tryEmitAbsolute(mlir::Type destTy) {
  // If we're producing a pointer, this is easy.
  auto destPtrTy = mlir::cast<cir::PointerType>(destTy);
  return cgm.getBuilder().getConstPtrAttr(
      destPtrTy, value.getLValueOffset().getQuantity());
}

ConstantLValue
ConstantLValueEmitter::tryEmitBase(const APValue::LValueBase &base) {
  // Handle values.
  if (const ValueDecl *d = base.dyn_cast<const ValueDecl *>()) {
    // The constant always points to the canonical declaration. We want to look
    // at properties of the most recent declaration at the point of emission.
    d = cast<ValueDecl>(d->getMostRecentDecl());

    if (d->hasAttr<WeakRefAttr>()) {
      cgm.errorNYI(d->getSourceRange(),
                   "ConstantLValueEmitter: emit pointer base for weakref");
      return {};
    }

    if (auto *fd = dyn_cast<FunctionDecl>(d)) {
      cgm.errorNYI(fd->getSourceRange(),
                   "ConstantLValueEmitter: function decl");
      return {};
    }

    if (auto *vd = dyn_cast<VarDecl>(d)) {
      cgm.errorNYI(vd->getSourceRange(), "ConstantLValueEmitter: var decl");
      return {};
    }
  }

  // Handle typeid(T).
  if (base.dyn_cast<TypeInfoLValue>()) {
    cgm.errorNYI("ConstantLValueEmitter: typeid");
    return {};
  }

  // Otherwise, it must be an expression.
  return Visit(base.get<const Expr *>());
}

ConstantLValue ConstantLValueEmitter::VisitConstantExpr(const ConstantExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: constant expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitCompoundLiteralExpr(const CompoundLiteralExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: compound literal");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitStringLiteral(const StringLiteral *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: string literal");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitObjCEncodeExpr(const ObjCEncodeExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: objc encode expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitObjCStringLiteral(const ObjCStringLiteral *e) {
  cgm.errorNYI(e->getSourceRange(),
               "ConstantLValueEmitter: objc string literal");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitObjCBoxedExpr(const ObjCBoxedExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: objc boxed expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitPredefinedExpr(const PredefinedExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: predefined expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitAddrLabelExpr(const AddrLabelExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: addr label expr");
  return {};
}

ConstantLValue ConstantLValueEmitter::VisitCallExpr(const CallExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: call expr");
  return {};
}

ConstantLValue ConstantLValueEmitter::VisitBlockExpr(const BlockExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: block expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitCXXTypeidExpr(const CXXTypeidExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: cxx typeid expr");
  return {};
}

ConstantLValue ConstantLValueEmitter::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *e) {
  cgm.errorNYI(e->getSourceRange(),
               "ConstantLValueEmitter: materialize temporary expr");
  return {};
}

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const VarDecl &d) {
  initializeNonAbstract();
  return markIfFailed(tryEmitPrivateForVarInit(d));
}

void ConstantEmitter::finalize(cir::GlobalOp gv) {
  assert(initializedNonAbstract &&
         "finalizing emitter that was used for abstract emission?");
  assert(!finalized && "finalizing emitter multiple times");
  assert(!gv.isDeclaration());
#ifndef NDEBUG
  // Note that we might also be Failed.
  finalized = true;
#endif // NDEBUG
}

mlir::Attribute
ConstantEmitter::tryEmitAbstractForInitializer(const VarDecl &d) {
  AbstractStateRAII state(*this, true);
  return tryEmitPrivateForVarInit(d);
}

ConstantEmitter::~ConstantEmitter() {
  assert((!initializedNonAbstract || finalized || failed) &&
         "not finalized after being initialized for non-abstract emission");
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &d) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!d.hasLocalStorage()) {
    QualType ty = cgm.getASTContext().getBaseElementType(d.getType());
    if (ty->isRecordType()) {
      if (const auto *e = dyn_cast_or_null<CXXConstructExpr>(d.getInit())) {
        const CXXConstructorDecl *cd = e->getConstructor();
        // FIXME: we should probably model this more closely to C++ than
        // just emitting a global with zero init (mimic what we do for trivial
        // assignments and whatnots). Since this is for globals shouldn't
        // be a problem for the near future.
        if (cd->isTrivial() && cd->isDefaultConstructor()) {
          const auto *cxxrd =
              cast<CXXRecordDecl>(ty->getAs<RecordType>()->getOriginalDecl())
                  ->getDefinitionOrSelf();
          if (cxxrd->getNumBases() != 0) {
            // There may not be anything additional to do here, but this will
            // force us to pause and test this path when it is supported.
            cgm.errorNYI("tryEmitPrivateForVarInit: cxx record with bases");
            return {};
          }
          if (!cgm.getTypes().isZeroInitializable(cxxrd)) {
            // To handle this case, we really need to go through
            // emitNullConstant, but we need an attribute, not a value
            cgm.errorNYI(
                "tryEmitPrivateForVarInit: non-zero-initializable cxx record");
            return {};
          }
          return cir::ZeroAttr::get(cgm.convertType(d.getType()));
        }
      }
    }
  }
  inConstantContext = d.hasConstantInitialization();

  const Expr *e = d.getInit();
  assert(e && "No initializer to emit");

  QualType destType = d.getType();

  if (!destType->isReferenceType()) {
    QualType nonMemoryDestType = getNonMemoryType(cgm, destType);
    if (mlir::Attribute c = ConstExprEmitter(*this).Visit(const_cast<Expr *>(e),
                                                          nonMemoryDestType))
      return emitForMemory(c, destType);
  }

  // Try to emit the initializer.  Note that this can allow some things that
  // are not allowed by tryEmitPrivateForMemory alone.
  if (APValue *value = d.evaluateValue())
    return tryEmitPrivateForMemory(*value, destType);

  return {};
}

mlir::Attribute ConstantEmitter::tryEmitConstantExpr(const ConstantExpr *ce) {
  if (!ce->hasAPValueResult())
    return {};

  QualType retType = ce->getType();
  if (ce->isGLValue())
    retType = cgm.getASTContext().getLValueReferenceType(retType);

  return emitAbstract(ce->getBeginLoc(), ce->getAPValueResult(), retType);
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForMemory(const APValue &value,
                                                         QualType destType) {
  QualType nonMemoryDestType = getNonMemoryType(cgm, destType);
  mlir::Attribute c = tryEmitPrivate(value, nonMemoryDestType);
  return (c ? emitForMemory(c, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::emitAbstract(SourceLocation loc,
                                              const APValue &value,
                                              QualType destType) {
  AbstractStateRAII state(*this, true);
  mlir::Attribute c = tryEmitPrivate(value, destType);
  if (!c)
    cgm.errorNYI(loc, "emitAbstract failed, emit null constaant");
  return c;
}

mlir::Attribute ConstantEmitter::emitForMemory(mlir::Attribute c,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (destType->getAs<AtomicType>()) {
    cgm.errorNYI("emitForMemory: atomic type");
    return {};
  }

  return c;
}

mlir::Attribute ConstantEmitter::tryEmitPrivate(const APValue &value,
                                                QualType destType) {
  auto &builder = cgm.getBuilder();
  switch (value.getKind()) {
  case APValue::None:
  case APValue::Indeterminate:
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate none or indeterminate");
    return {};
  case APValue::Int: {
    mlir::Type ty = cgm.convertType(destType);
    if (mlir::isa<cir::BoolType>(ty))
      return builder.getCIRBoolAttr(value.getInt().getZExtValue());
    assert(mlir::isa<cir::IntType>(ty) && "expected integral type");
    return cir::IntAttr::get(ty, value.getInt());
  }
  case APValue::Float: {
    const llvm::APFloat &init = value.getFloat();
    if (&init.getSemantics() == &llvm::APFloat::IEEEhalf() &&
        !cgm.getASTContext().getLangOpts().NativeHalfType &&
        cgm.getASTContext().getTargetInfo().useFP16ConversionIntrinsics()) {
      cgm.errorNYI("ConstExprEmitter::tryEmitPrivate half");
      return {};
    }

    mlir::Type ty = cgm.convertType(destType);
    assert(mlir::isa<cir::FPTypeInterface>(ty) &&
           "expected floating-point type");
    return cir::FPAttr::get(ty, init);
  }
  case APValue::Array: {
    const ArrayType *arrayTy = cgm.getASTContext().getAsArrayType(destType);
    const QualType arrayElementTy = arrayTy->getElementType();
    const unsigned numElements = value.getArraySize();
    const unsigned numInitElts = value.getArrayInitializedElts();

    mlir::Attribute filler;
    if (value.hasArrayFiller()) {
      filler = tryEmitPrivate(value.getArrayFiller(), arrayElementTy);
      if (!filler)
        return {};
    }

    SmallVector<mlir::TypedAttr, 16> elements;
    if (filler && builder.isNullValue(filler))
      elements.reserve(numInitElts + 1);
    else
      elements.reserve(numInitElts);

    mlir::Type commonElementType;
    for (unsigned i = 0; i < numInitElts; ++i) {
      const APValue &arrayElement = value.getArrayInitializedElt(i);
      const mlir::Attribute element =
          tryEmitPrivateForMemory(arrayElement, arrayElementTy);
      if (!element)
        return {};

      const mlir::TypedAttr elementTyped = mlir::cast<mlir::TypedAttr>(element);
      if (i == 0)
        commonElementType = elementTyped.getType();
      else if (elementTyped.getType() != commonElementType) {
        commonElementType = {};
      }

      elements.push_back(elementTyped);
    }

    mlir::TypedAttr typedFiller = llvm::cast_or_null<mlir::TypedAttr>(filler);
    if (filler && !typedFiller)
      cgm.errorNYI("array filler should always be typed");

    mlir::Type desiredType = cgm.convertType(destType);
    return emitArrayConstant(cgm, desiredType, commonElementType, numElements,
                             elements, typedFiller);
  }
  case APValue::Vector: {
    const QualType elementType =
        destType->castAs<VectorType>()->getElementType();
    const unsigned numElements = value.getVectorLength();

    SmallVector<mlir::Attribute, 16> elements;
    elements.reserve(numElements);

    for (unsigned i = 0; i < numElements; ++i) {
      const mlir::Attribute element =
          tryEmitPrivateForMemory(value.getVectorElt(i), elementType);
      if (!element)
        return {};
      elements.push_back(element);
    }

    const auto desiredVecTy =
        mlir::cast<cir::VectorType>(cgm.convertType(destType));

    return cir::ConstVectorAttr::get(
        desiredVecTy,
        mlir::ArrayAttr::get(cgm.getBuilder().getContext(), elements));
  }
  case APValue::MemberPointer: {
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate member pointer");
    return {};
  }
  case APValue::LValue:
    return ConstantLValueEmitter(*this, value, destType).tryEmit();
  case APValue::Struct:
  case APValue::Union:
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate struct or union");
    return {};
  case APValue::ComplexInt:
  case APValue::ComplexFloat: {
    mlir::Type desiredType = cgm.convertType(destType);
    cir::ComplexType complexType =
        mlir::dyn_cast<cir::ComplexType>(desiredType);

    mlir::Type complexElemTy = complexType.getElementType();
    if (isa<cir::IntType>(complexElemTy)) {
      llvm::APSInt real = value.getComplexIntReal();
      llvm::APSInt imag = value.getComplexIntImag();
      return builder.getAttr<cir::ConstComplexAttr>(
          complexType, cir::IntAttr::get(complexElemTy, real),
          cir::IntAttr::get(complexElemTy, imag));
    }

    assert(isa<cir::FPTypeInterface>(complexElemTy) &&
           "expected floating-point type");
    llvm::APFloat real = value.getComplexFloatReal();
    llvm::APFloat imag = value.getComplexFloatImag();
    return builder.getAttr<cir::ConstComplexAttr>(
        complexType, cir::FPAttr::get(complexElemTy, real),
        cir::FPAttr::get(complexElemTy, imag));
  }
  case APValue::FixedPoint:
  case APValue::AddrLabelDiff:
    cgm.errorNYI(
        "ConstExprEmitter::tryEmitPrivate fixed point, addr label diff");
    return {};
  }
  llvm_unreachable("Unknown APValue kind");
}

mlir::Value CIRGenModule::emitNullConstant(QualType t, mlir::Location loc) {
  if (t->getAs<PointerType>()) {
    return builder.getNullPtr(getTypes().convertTypeForMem(t), loc);
  }

  if (getTypes().isZeroInitializable(t))
    return builder.getNullValue(getTypes().convertTypeForMem(t), loc);

  if (getASTContext().getAsConstantArrayType(t)) {
    errorNYI("CIRGenModule::emitNullConstant ConstantArrayType");
  }

  if (t->getAs<RecordType>())
    errorNYI("CIRGenModule::emitNullConstant RecordType");

  assert(t->isMemberDataPointerType() &&
         "Should only see pointers to data members here!");

  errorNYI("CIRGenModule::emitNullConstant unsupported type");
  return {};
}
