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
    : public StmtVisitor<ConstExprEmitter, mlir::Value, QualType> {
  CIRGenModule &CGM;
  LLVM_ATTRIBUTE_UNUSED ConstantEmitter &Emitter;

public:
  ConstExprEmitter(ConstantEmitter &emitter)
      : CGM(emitter.CGM), Emitter(emitter) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value VisitStmt(Stmt *S, QualType T) { return nullptr; }

  mlir::Value VisitConstantExpr(ConstantExpr *CE, QualType T) {
    assert(0 && "unimplemented");
    // if (mlir::Value Result = Emitter.tryEmitConstantExpr(CE))
    //   return Result;
    // return Visit(CE->getSubExpr(), T);
    return {};
  }

  mlir::Value VisitParenExpr(ParenExpr *PE, QualType T) {
    return Visit(PE->getSubExpr(), T);
  }

  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *PE,
                                    QualType T) {
    return Visit(PE->getReplacement(), T);
  }

  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *GE, QualType T) {
    return Visit(GE->getResultExpr(), T);
  }

  mlir::Value VisitChooseExpr(ChooseExpr *CE, QualType T) {
    return Visit(CE->getChosenSubExpr(), T);
  }

  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *E, QualType T) {
    return Visit(E->getInitializer(), T);
  }

  mlir::Value VisitCastExpr(CastExpr *E, QualType destType) {
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

  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE, QualType T) {
    // TODO(cir): figure out CIR story here...
    // No need for a DefaultInitExprScope: we don't handle 'this' in a
    // constant expression.
    return Visit(DIE->getExpr(), T);
  }

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *E, QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  mlir::Value VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E,
                                            QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  mlir::Value EmitArrayInitialization(InitListExpr *ILE, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Value EmitRecordInitialization(InitListExpr *ILE, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Value VisitImplicitValueInitExpr(ImplicitValueInitExpr *E, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Value VisitInitListExpr(InitListExpr *ILE, QualType T) {
    if (ILE->isTransparent())
      return Visit(ILE->getInit(0), T);

    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE, T);

    if (ILE->getType()->isRecordType())
      return EmitRecordInitialization(ILE, T);

    return nullptr;
  }

  mlir::Value VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E,
                                            QualType destType) {
    auto C = Visit(E->getBase(), destType);
    if (!C)
      return nullptr;

    assert(0 && "not implemented");
    return {};
  }

  mlir::Value VisitCXXConstructExpr(CXXConstructExpr *E, QualType Ty) {
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

  mlir::Value VisitStringLiteral(StringLiteral *E, QualType T) {
    // This is a string literal initializing an array in an initializer.
    assert(0 && "not implemented");
    return {};
  }

  mlir::Value VisitObjCEncodeExpr(ObjCEncodeExpr *E, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Value VisitUnaryExtension(const UnaryOperator *E, QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  // Utility methods
  mlir::Type ConvertType(QualType T) { return CGM.getTypes().ConvertType(T); }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Value ConstantEmitter::tryEmitForInitializer(const VarDecl &D) {
  initializeNonAbstract(D.getType().getAddressSpace());
  return markIfFailed(tryEmitPrivateForVarInit(D));
}

mlir::Value ConstantEmitter::tryEmitForInitializer(const Expr *E,
                                                   LangAS destAddrSpace,
                                                   QualType destType) {
  initializeNonAbstract(destAddrSpace);
  return markIfFailed(tryEmitPrivateForMemory(E, destType));
}

// mlir::Value ConstantEmitter::emitForInitializer(const APValue &value,
//                                                 LangAS destAddrSpace,
//                                                 QualType destType) {
//   initializeNonAbstract(destAddrSpace);
//   auto C = tryEmitPrivateForMemory(value, destType);
//   assert(C && "couldn't emit constant value non-abstractly?");
//   return C;
// }

// void ConstantEmitter::finalize(llvm::GlobalVariable *global) {
//   assert(InitializedNonAbstract &&
//          "finalizing emitter that was used for abstract emission?");
//   assert(!Finalized && "finalizing emitter multiple times");
//   assert(global->getInitializer());

//   // Note that we might also be Failed.
//   Finalized = true;

//   if (!PlaceholderAddresses.empty()) {
//     assert(0 && "not implemented");
//   }
// }

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

mlir::Value ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &D) {
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

mlir::Value ConstantEmitter::tryEmitPrivateForMemory(const APValue &value,
                                                     QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitPrivate(value, nonMemoryDestType);
  return (C ? emitForMemory(C, destType) : nullptr);
}

mlir::Value ConstantEmitter::tryEmitPrivateForMemory(const clang::Expr *E,
                                                     clang::QualType T) {
  llvm_unreachable("NYI");
}

mlir::Value ConstantEmitter::emitForMemory(CIRGenModule &CGM, mlir::Value C,
                                           QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (auto AT = destType->getAs<AtomicType>()) {
    assert(0 && "not implemented");
  }

  // Zero-extend bool.
  if (C.getType().isa<mlir::cir::BoolType>()) {
    assert(0 && "not implemented");
  }

  return C;
}

mlir::Value ConstantEmitter::tryEmitPrivate(const APValue &Value,
                                            QualType DestType) {
  switch (Value.getKind()) {
  case APValue::None:
  case APValue::Indeterminate:
    // TODO(cir): LLVM models out-of-lifetime and indeterminate values as
    // 'undef'. Find out what's better for CIR.
    assert(0 && "not implemented");
  case APValue::Int:
    assert(0 && "not implemented");
  case APValue::LValue:
  case APValue::FixedPoint:
  case APValue::ComplexInt:
  case APValue::Float:
  case APValue::ComplexFloat:
  case APValue::Vector:
  case APValue::AddrLabelDiff:
  case APValue::Struct:
  case APValue::Union:
  case APValue::Array:
  case APValue::MemberPointer:
    assert(0 && "not implemented");
  }
  llvm_unreachable("Unknown APValue kind");
}
