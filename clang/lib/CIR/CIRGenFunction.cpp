//===- CIRGenFunction.cpp - Emit CIR from ASTs for a Function -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/Basic/TargetInfo.h"

using namespace cir;
using namespace clang;

CIRGenFunction::CIRGenFunction(CIRGenModule &CGM)
    : CGM{CGM}, SanOpts(CGM.getLangOpts().Sanitize) {}

clang::ASTContext &CIRGenFunction::getContext() const {
  return CGM.getASTContext();
}

TypeEvaluationKind CIRGenFunction::getEvaluationKind(QualType type) {
  type = type.getCanonicalType();
  while (true) {
    switch (type->getTypeClass()) {
#define TYPE(name, parent)
#define ABSTRACT_TYPE(name, parent)
#define NON_CANONICAL_TYPE(name, parent) case Type::name:
#define DEPENDENT_TYPE(name, parent) case Type::name:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(name, parent) case Type::name:
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("non-canonical or dependent type in IR-generation");

    case Type::ArrayParameter:
      llvm_unreachable("NYI");

    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      llvm_unreachable("undeduced type in IR-generation");

    // Various scalar types.
    case Type::Builtin:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Enum:
    case Type::ObjCObjectPointer:
    case Type::Pipe:
    case Type::BitInt:
      return TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

static bool hasInAllocaArgs(CIRGenModule &CGM, CallingConv ExplicitCC,
                            ArrayRef<QualType> ArgTypes) {
  assert(ExplicitCC != CC_Swift && ExplicitCC != CC_SwiftAsync && "Swift NYI");
  assert(!CGM.getTarget().getCXXABI().isMicrosoft() && "MSABI NYI");

  return false;
}

void CIRGenFunction::buildCallArgs(
    CallArgList &Args, PrototypeWrapper Prototype,
    llvm::iterator_range<CallExpr::const_arg_iterator> ArgRange,
    AbstractCallee AC, unsigned ParamsToSkip, EvaluationOrder Order) {

  llvm::SmallVector<QualType, 16> ArgTypes;

  assert((ParamsToSkip == 0 || Prototype.P) &&
         "Can't skip parameters if type info is not provided");

  // This variable only captures *explicitly* written conventions, not those
  // applied by default via command line flags or target defaults, such as
  // thiscall, appcs, stdcall via -mrtd, etc. Computing that correctly would
  // require knowing if this is a C++ instance method or being able to see
  // unprotyped FunctionTypes.
  CallingConv ExplicitCC = CC_C;

  // First, if a prototype was provided, use those argument types.
  bool IsVariadic = false;
  if (Prototype.P) {
    const auto *MD = Prototype.P.dyn_cast<const ObjCMethodDecl *>();
    assert(!MD && "ObjCMethodDecl NYI");

    const auto *FPT = Prototype.P.get<const FunctionProtoType *>();
    IsVariadic = FPT->isVariadic();
    assert(!IsVariadic && "Variadic functions NYI");
    ExplicitCC = FPT->getExtInfo().getCC();
    ArgTypes.assign(FPT->param_type_begin() + ParamsToSkip,
                    FPT->param_type_end());
  }

  // If we still have any arguments, emit them using the type of the argument.
  for (auto *A : llvm::drop_begin(ArgRange, ArgTypes.size())) {
    assert(!IsVariadic && "Variadic functions NYI");
    ArgTypes.push_back(A->getType());
  };
  assert((int)ArgTypes.size() == (ArgRange.end() - ArgRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI, because
  // arguments are destroyed left to right in the callee. As a special case,
  // there are certain language constructs taht require left-to-right
  // evaluation, and in those cases we consider the evaluation order requirement
  // to trump the "destruction order is reverse construction order" guarantee.
  bool LeftToRight = true;
  assert(!CGM.getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee() &&
         "MSABI NYI");
  assert(!hasInAllocaArgs(CGM, ExplicitCC, ArgTypes) && "NYI");

  // Evaluate each argument in the appropriate order.
  size_t CallArgsStart = Args.size();
  assert(ArgTypes.size() == 0 && "Args NYI");

  if (!LeftToRight) {
    // Un-reverse the arguments we just evaluated so they match up with the CIR
    // function.
    std::reverse(Args.begin() + CallArgsStart, Args.end());
  }
}

/// Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
/// TODO: if this is an aggregate expression, add a AggValueSlot to indicate
/// where the result should be returned.
RValue CIRGenFunction::buildAnyExpr(const Expr *E) {
  switch (CIRGenFunction::getEvaluationKind(E->getType())) {
  case TEK_Scalar:
    return RValue::get(CGM.buildScalarExpr(E));
  case TEK_Complex:
    assert(0 && "not implemented");
  case TEK_Aggregate:
    assert(0 && "not implemented");
  }
  llvm_unreachable("bad evaluation kind");
}
