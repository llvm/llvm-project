//===---- CIRGenExprCst.cpp - Emit LLVM Code from Constant Expressions ----===//
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

#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

using namespace clang;
using namespace cir;

//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//

namespace {

// This class only needs to handle arrays, structs and unions.
//
// In LLVM codegen, when outside C++11 mode, those types are not constant
// folded, while all other types are handled by constant folding.
//
// In CIR codegen, instead of folding things here, we should defer that work
// to MLIR: do not attempt to do much here.
class ConstExprEmitter
    : public StmtVisitor<ConstExprEmitter, mlir::Attribute, QualType> {
  CIRGenModule &CGM;
  LLVM_ATTRIBUTE_UNUSED ConstantEmitter &Emitter;

public:
  ConstExprEmitter(ConstantEmitter &emitter)
      : CGM(emitter.CGM), Emitter(emitter) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Attribute VisitStmt(Stmt *S, QualType T) { return nullptr; }

  mlir::Attribute VisitConstantExpr(ConstantExpr *CE, QualType T) {
    assert(0 && "unimplemented");
    return {};
  }

  mlir::Attribute VisitParenExpr(ParenExpr *PE, QualType T) {
    return Visit(PE->getSubExpr(), T);
  }

  mlir::Attribute
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *PE,
                                    QualType T) {
    return Visit(PE->getReplacement(), T);
  }

  mlir::Attribute VisitGenericSelectionExpr(GenericSelectionExpr *GE,
                                            QualType T) {
    return Visit(GE->getResultExpr(), T);
  }

  mlir::Attribute VisitChooseExpr(ChooseExpr *CE, QualType T) {
    return Visit(CE->getChosenSubExpr(), T);
  }

  mlir::Attribute VisitCompoundLiteralExpr(CompoundLiteralExpr *E, QualType T) {
    return Visit(E->getInitializer(), T);
  }

  mlir::Attribute VisitCastExpr(CastExpr *E, QualType destType) {
    if (const auto *ECE = dyn_cast<ExplicitCastExpr>(E))
      assert(0 && "not implemented");
    Expr *subExpr = E->getSubExpr();

    switch (E->getCastKind()) {
    case CK_HLSLArrayRValue:
    case CK_HLSLVectorTruncation:
    case CK_ToUnion: {
      assert(0 && "not implemented");
    }

    case CK_AddressSpaceConversion: {
      assert(0 && "not implemented");
    }

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

    case CK_ReinterpretMemberPointer:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer: {
      assert(0 && "not implemented");
    }

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_ARCProduceObject:
    case CK_ARCConsumeObject:
    case CK_ARCReclaimReturnedObject:
    case CK_ARCExtendBlockObject:
    case CK_CopyAndAutoreleaseBlockObject:
      return nullptr;

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
      return nullptr;
    }
    llvm_unreachable("Invalid CastKind");
  }

  mlir::Attribute VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE, QualType T) {
    // TODO(cir): figure out CIR story here...
    // No need for a DefaultInitExprScope: we don't handle 'this' in a
    // constant expression.
    return Visit(DIE->getExpr(), T);
  }

  mlir::Attribute VisitExprWithCleanups(ExprWithCleanups *E, QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  mlir::Attribute VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E,
                                                QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  mlir::Attribute EmitArrayInitialization(InitListExpr *ILE, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute EmitRecordInitialization(InitListExpr *ILE, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitImplicitValueInitExpr(ImplicitValueInitExpr *E,
                                             QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitInitListExpr(InitListExpr *ILE, QualType T) {
    if (ILE->isTransparent())
      return Visit(ILE->getInit(0), T);

    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE, T);

    if (ILE->getType()->isRecordType())
      return EmitRecordInitialization(ILE, T);

    return nullptr;
  }

  mlir::Attribute VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E,
                                                QualType destType) {
    auto C = Visit(E->getBase(), destType);
    if (!C)
      return nullptr;

    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitCXXConstructExpr(CXXConstructExpr *E, QualType Ty) {
    if (!E->getConstructor()->isTrivial())
      return nullptr;

    // Only default and copy/move constructors can be trivial.
    if (E->getNumArgs()) {
      assert(E->getNumArgs() == 1 && "trivial ctor with > 1 argument");
      assert(E->getConstructor()->isCopyOrMoveConstructor() &&
             "trivial ctor has argument but isn't a copy/move ctor");

      Expr *Arg = E->getArg(0);
      assert(CGM.getASTContext().hasSameUnqualifiedType(Ty, Arg->getType()) &&
             "argument to copy ctor is of wrong type");

      return Visit(Arg, Ty);
    }

    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitStringLiteral(StringLiteral *E, QualType T) {
    // This is a string literal initializing an array in an initializer.
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitObjCEncodeExpr(ObjCEncodeExpr *E, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitUnaryExtension(const UnaryOperator *E, QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  // Utility methods
  mlir::Type ConvertType(QualType T) { return CGM.getTypes().ConvertType(T); }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute ConstantEmitter::validateAndPopAbstract(mlir::Attribute C,
                                                        AbstractState saved) {
  Abstract = saved.OldValue;

  assert(saved.OldPlaceholdersSize == PlaceholderAddresses.size() &&
         "created a placeholder while doing an abstract emission?");

  // No validation necessary for now.
  // No cleanup to do for now.
  return C;
}

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const VarDecl &D) {
  initializeNonAbstract(D.getType().getAddressSpace());
  return markIfFailed(tryEmitPrivateForVarInit(D));
}

void ConstantEmitter::finalize(mlir::cir::GlobalOp global) {
  assert(InitializedNonAbstract &&
         "finalizing emitter that was used for abstract emission?");
  assert(!Finalized && "finalizing emitter multiple times");
  assert(!global.isDeclaration());

  // Note that we might also be Failed.
  Finalized = true;

  if (!PlaceholderAddresses.empty()) {
    assert(0 && "not implemented");
  }
}

ConstantEmitter::~ConstantEmitter() {
  assert((!InitializedNonAbstract || Finalized || Failed) &&
         "not finalized after being initialized for non-abstract emission");
  assert(PlaceholderAddresses.empty() && "unhandled placeholders");
}

// TODO(cir): this can be shared with LLVM's codegen
static QualType getNonMemoryType(CIRGenModule &CGM, QualType type) {
  if (auto AT = type->getAs<AtomicType>()) {
    return CGM.getASTContext().getQualifiedType(AT->getValueType(),
                                                type.getQualifiers());
  }
  return type;
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &D) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!D.hasLocalStorage()) {
    QualType Ty = CGM.getASTContext().getBaseElementType(D.getType());
    if (Ty->isRecordType())
      if (const CXXConstructExpr *E =
              dyn_cast_or_null<CXXConstructExpr>(D.getInit())) {
        const CXXConstructorDecl *CD = E->getConstructor();
        if (CD->isTrivial() && CD->isDefaultConstructor())
          assert(0 && "not implemented");
      }
  }
  InConstantContext = D.hasConstantInitialization();

  QualType destType = D.getType();

  // Try to emit the initializer.  Note that this can allow some things that
  // are not allowed by tryEmitPrivateForMemory alone.
  if (auto value = D.evaluateValue()) {
    return tryEmitPrivateForMemory(*value, destType);
  }

  assert(0 && "not implemented");
  return {};
}

mlir::Attribute ConstantEmitter::tryEmitAbstract(const APValue &value,
                                                 QualType destType) {
  auto state = pushAbstract();
  auto C = tryEmitPrivate(value, destType);
  return validateAndPopAbstract(C, state);
}

mlir::Attribute ConstantEmitter::tryEmitAbstractForMemory(const APValue &value,
                                                          QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitAbstract(value, nonMemoryDestType);
  return (C ? emitForMemory(C, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForMemory(const APValue &value,
                                                         QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitPrivate(value, nonMemoryDestType);
  if (C) {
    auto attr = emitForMemory(C, destType);
    auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(attr);
    if (!typedAttr)
      llvm_unreachable("this should always be typed");
    return typedAttr;
  }

  return nullptr;
}

mlir::Attribute ConstantEmitter::emitForMemory(CIRGenModule &CGM,
                                               mlir::Attribute C,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (auto AT = destType->getAs<AtomicType>()) {
    assert(0 && "not implemented");
  }

  // Zero-extend bool.
  auto typed = C.dyn_cast<mlir::TypedAttr>();
  if (typed && typed.getType().isa<mlir::cir::BoolType>()) {
    assert(0 && "not implemented");
  }

  return C;
}

static mlir::Attribute
buildArrayConstant(CIRGenModule &CGM, mlir::Type DesiredType,
                   mlir::Type CommonElementType, unsigned ArrayBound,
                   SmallVectorImpl<mlir::TypedAttr> &Elements,
                   mlir::TypedAttr Filler) {
  auto isNullValue = [&](mlir::Attribute f) {
    // TODO(cir): introduce char type in CIR and check for that instead.
    auto intVal = f.dyn_cast_or_null<mlir::IntegerAttr>();
    assert(intVal && "not implemented");
    if (intVal.getInt() == 0)
      return true;
    return false;
  };

  // Figure out how long the initial prefix of non-zero elements is.
  unsigned NonzeroLength = ArrayBound;
  if (Elements.size() < NonzeroLength && isNullValue(Filler))
    NonzeroLength = Elements.size();
  if (NonzeroLength == Elements.size()) {
    while (NonzeroLength > 0 && isNullValue(Elements[NonzeroLength - 1]))
      --NonzeroLength;
  }

  if (NonzeroLength == 0)
    assert(0 && "NYE");

  // Add a zeroinitializer array filler if we have lots of trailing zeroes.
  unsigned TrailingZeroes = ArrayBound - NonzeroLength;
  if (TrailingZeroes >= 8) {
    assert(0 && "NYE");
    assert(Elements.size() >= NonzeroLength &&
           "missing initializer for non-zero element");

    // TODO(cir): If all the elements had the same type up to the trailing
    // zeroes, emit a struct of two arrays (the nonzero data and the
    // zeroinitializer). Use DesiredType to get the element type.
  } else if (Elements.size() != ArrayBound) {
    // Otherwise pad to the right size with the filler if necessary.
    Elements.resize(ArrayBound, Filler);
    if (Filler.getType() != CommonElementType)
      CommonElementType = {};
  }

  // If all elements have the same type, just emit an array constant.
  if (CommonElementType) {
    SmallVector<mlir::Attribute, 4> Eles;
    Eles.reserve(Elements.size());
    for (auto const &Element : Elements)
      Eles.push_back(Element);

    return mlir::cir::CstArrayAttr::get(
        mlir::cir::ArrayType::get(CGM.getBuilder().getContext(),
                                  CommonElementType, ArrayBound),
        mlir::ArrayAttr::get(CGM.getBuilder().getContext(), Eles));
  }

  // We have mixed types. Use a packed struct.
  assert(0 && "NYE");
  return {};
}

mlir::Attribute ConstantEmitter::tryEmitPrivate(const APValue &Value,
                                                QualType DestType) {
  switch (Value.getKind()) {
  case APValue::None:
  case APValue::Indeterminate:
    // TODO(cir): LLVM models out-of-lifetime and indeterminate values as
    // 'undef'. Find out what's better for CIR.
    assert(0 && "not implemented");
  case APValue::Int: {
    mlir::Type ty = CGM.getCIRType(DestType);
    return CGM.getBuilder().getIntegerAttr(ty, Value.getInt());
  }
  case APValue::Float: {
    const llvm::APFloat &Init = Value.getFloat();
    if (&Init.getSemantics() == &llvm::APFloat::IEEEhalf() &&
        !CGM.getASTContext().getLangOpts().NativeHalfType &&
        CGM.getASTContext().getTargetInfo().useFP16ConversionIntrinsics())
      assert(0 && "not implemented");
    else {
      mlir::Type ty = CGM.getCIRType(DestType);
      return CGM.getBuilder().getFloatAttr(ty, Init);
    }
  }
  case APValue::Array: {
    const ArrayType *ArrayTy = CGM.getASTContext().getAsArrayType(DestType);
    unsigned NumElements = Value.getArraySize();
    unsigned NumInitElts = Value.getArrayInitializedElts();
    auto isNullValue = [&](mlir::Attribute f) {
      // TODO(cir): introduce char type in CIR and check for that instead.
      auto intVal = f.dyn_cast_or_null<mlir::IntegerAttr>();
      assert(intVal && "not implemented");
      if (intVal.getInt() == 0)
        return true;
      return false;
    };

    // Emit array filler, if there is one.
    mlir::Attribute Filler;
    if (Value.hasArrayFiller()) {
      Filler = tryEmitAbstractForMemory(Value.getArrayFiller(),
                                        ArrayTy->getElementType());
      if (!Filler)
        return {};
    }

    // Emit initializer elements.
    SmallVector<mlir::TypedAttr, 16> Elts;
    if (Filler && isNullValue(Filler))
      Elts.reserve(NumInitElts + 1);
    else
      Elts.reserve(NumElements);

    mlir::Type CommonElementType;
    for (unsigned I = 0; I < NumInitElts; ++I) {
      auto C = tryEmitPrivateForMemory(Value.getArrayInitializedElt(I),
                                       ArrayTy->getElementType());
      if (!C)
        return {};

      assert(C.isa<mlir::TypedAttr>() && "This should always be a TypedAttr.");
      auto CTyped = C.cast<mlir::TypedAttr>();

      if (I == 0)
        CommonElementType = CTyped.getType();
      else if (CTyped.getType() != CommonElementType)
        CommonElementType = {};
      auto typedC = llvm::dyn_cast<mlir::TypedAttr>(C);
      if (!typedC)
        llvm_unreachable("this should always be typed");
      Elts.push_back(typedC);
    }

    auto Desired = CGM.getTypes().ConvertType(DestType);

    auto typedFiller = llvm::dyn_cast_or_null<mlir::TypedAttr>(Filler);
    if (Filler && !typedFiller)
      llvm_unreachable("this should always be typed");

    return buildArrayConstant(CGM, Desired, CommonElementType, NumElements,
                              Elts, typedFiller);
  }
  case APValue::LValue:
  case APValue::FixedPoint:
  case APValue::ComplexInt:
  case APValue::ComplexFloat:
  case APValue::Vector:
  case APValue::AddrLabelDiff:
  case APValue::Struct:
  case APValue::Union:
  case APValue::MemberPointer:
    assert(0 && "not implemented");
  }
  llvm_unreachable("Unknown APValue kind");
}
