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
    cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitCastExpr");
    return {};
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
    cgm.errorNYI(ile->getBeginLoc(), "ConstExprEmitter::VisitInitListExpr");
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
    cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitStringLiteral");
    return {};
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
    return cir::ZeroAttr::get(builder.getContext(), desiredType);

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
        cir::ArrayType::get(builder.getContext(), commonElementType,
                            arrayBound),
        mlir::ArrayAttr::get(builder.getContext(), eles));
  }

  cgm.errorNYI("array with different type elements");
  return {};
}

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute
ConstantEmitter::tryEmitAbstractForInitializer(const VarDecl &d) {
  AbstractStateRAII state(*this, true);
  return tryEmitPrivateForVarInit(d);
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &d) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!d.hasLocalStorage()) {
    QualType ty = cgm.getASTContext().getBaseElementType(d.getType());
    if (ty->isRecordType())
      if (d.getInit() && isa<CXXConstructExpr>(d.getInit())) {
        cgm.errorNYI(d.getInit()->getBeginLoc(),
                     "tryEmitPrivateForVarInit CXXConstructExpr");
        return {};
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
    return cgm.getBuilder().getAttr<cir::IntAttr>(ty, value.getInt());
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
    assert(mlir::isa<cir::CIRFPTypeInterface>(ty) &&
           "expected floating-point type");
    return cgm.getBuilder().getAttr<cir::FPAttr>(ty, init);
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
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate vector");
    return {};
  }
  case APValue::MemberPointer: {
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate member pointer");
    return {};
  }
  case APValue::LValue: {

    if (value.getLValueBase()) {
      cgm.errorNYI("non-null pointer initialization");
    } else {

      mlir::Type desiredType = cgm.convertType(destType);
      if (const cir::PointerType ptrType =
              mlir::dyn_cast<cir::PointerType>(desiredType)) {
        return builder.getConstPtrAttr(ptrType,
                                       value.getLValueOffset().getQuantity());
      } else {
        llvm_unreachable("non-pointer variable initialized with a pointer");
      }
    }
    return {};
  }
  case APValue::Struct:
  case APValue::Union:
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate struct or union");
    return {};
  case APValue::FixedPoint:
  case APValue::ComplexInt:
  case APValue::ComplexFloat:
  case APValue::AddrLabelDiff:
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate fixed point, complex int, "
                 "complex float, addr label diff");
    return {};
  }
  llvm_unreachable("Unknown APValue kind");
}
