//===- CIROpsClassMembers.cpp - CIR ops in partition `ClassMembers` -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `ClassMembers`: the
// hand-written parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsClassMembers.h"
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h"
#include "clang/CIR/Dialect/IR/CIROpsSymbols.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// ArrayCtor & ArrayDtor
//===----------------------------------------------------------------------===//

template <typename Op> static LogicalResult verifyArrayCtorDtor(Op op) {
  auto ptrTy = mlir::cast<cir::PointerType>(op.getAddr().getType());
  mlir::Type pointeeTy = ptrTy.getPointee();

  mlir::Block &body = op.getBody().front();
  if (body.getNumArguments() != 1)
    return op.emitOpError("body must have exactly one block argument");

  auto expectedEltPtrTy =
      mlir::dyn_cast<cir::PointerType>(body.getArgument(0).getType());
  if (!expectedEltPtrTy)
    return op.emitOpError("block argument must be a !cir.ptr type");

  if (op.getNumElements()) {
    auto recTy = mlir::dyn_cast<cir::RecordType>(pointeeTy);
    if (!recTy)
      return op.emitOpError(
          "when 'num_elements' is present, 'addr' must be a pointer to a "
          "!cir.struct or !cir.union type");

    if (expectedEltPtrTy != ptrTy)
      return op.emitOpError("when 'num_elements' is present, 'addr' type must "
                            "match the block argument type");
  } else {
    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(pointeeTy);
    if (!arrayTy)
      return op.emitOpError(
          "when 'num_elements' is absent, 'addr' must be a pointer to a "
          "!cir.array type");

    mlir::Type innerEltTy = arrayTy.getElementType();
    while (auto nested = mlir::dyn_cast<cir::ArrayType>(innerEltTy))
      innerEltTy = nested.getElementType();

    auto recTy = mlir::dyn_cast<cir::RecordType>(innerEltTy);
    if (!recTy)
      return op.emitOpError("the block argument type must be a pointer to a "
                            "!cir.struct or !cir.union type");

    if (expectedEltPtrTy.getPointee() != innerEltTy)
      return op.emitOpError(
          "block argument pointee type must match the innermost array "
          "element type");
  }

  return success();
}

LogicalResult cir::ArrayCtor::verify() {
  if (failed(verifyArrayCtorDtor(*this)))
    return failure();

  mlir::Region &partialDtor = getPartialDtor();
  if (!partialDtor.empty()) {
    mlir::Block &dtorBlock = partialDtor.front();
    if (dtorBlock.getNumArguments() != 1)
      return emitOpError("partial_dtor must have exactly one block argument");

    auto bodyArgTy = getBody().front().getArgument(0).getType();
    if (dtorBlock.getArgument(0).getType() != bodyArgTy)
      return emitOpError("partial_dtor block argument type must match "
                         "the body block argument type");
  }
  return success();
}
LogicalResult cir::ArrayDtor::verify() { return verifyArrayCtorDtor(*this); }

//===----------------------------------------------------------------------===//
// VTableAddrPointOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::VTableAddrPointOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  StringRef name = getName();

  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global.
  auto op =
      symbolTable.lookupNearestSymbolFrom<cir::GlobalOp>(*this, getNameAttr());
  if (!op)
    return emitOpError("'")
           << name << "' does not reference a valid cir.global";
  std::optional<mlir::Attribute> init = op.getInitialValue();
  if (!init)
    return success();
  if (!isa<cir::VTableAttr>(*init))
    return emitOpError("Expected #cir.vtable in initializer for global '")
           << name << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// VTTAddrPointOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::VTTAddrPointOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // VTT ptr is not coming from a symbol.
  if (!getName())
    return success();
  StringRef name = *getName();

  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global op.
  auto op =
      symbolTable.lookupNearestSymbolFrom<cir::GlobalOp>(*this, getNameAttr());
  if (!op)
    return emitOpError("'")
           << name << "' does not reference a valid cir.global";
  std::optional<mlir::Attribute> init = op.getInitialValue();
  if (!init)
    return success();
  if (!isa<cir::ConstArrayAttr>(*init))
    return emitOpError(
               "Expected constant array in initializer for global VTT '")
           << name << "'";
  return success();
}

LogicalResult cir::VTTAddrPointOp::verify() {
  // The operation uses either a symbol or a value to operate, but not both
  if (getName() && getSymAddr())
    return emitOpError("should use either a symbol or value, but not both");

  // If not a symbol, stick with the concrete type used for getSymAddr.
  if (getSymAddr())
    return success();

  mlir::Type resultType = getAddr().getType();
  mlir::Type resTy = cir::PointerType::get(
      cir::PointerType::get(cir::VoidType::get(getContext())));

  if (resultType != resTy)
    return emitOpError("result type must be ")
           << resTy << ", but provided result type is " << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// BaseClassAddrOp & DerivedClassAddrOp
//===----------------------------------------------------------------------===//

LogicalResult cir::BaseClassAddrOp::verify() {
  if (getSrcAddr().getType().getAddrSpace() != getType().getAddrSpace())
    return emitOpError() << "address space mismatch";
  return mlir::success();
}

LogicalResult cir::DerivedClassAddrOp::verify() {
  if (getSrcAddr().getType().getAddrSpace() != getType().getAddrSpace())
    return emitOpError() << "address space mismatch";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BaseDataMemberOp & DerivedDataMemberOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyMemberPtrCast(Operation *op, mlir::Value src,
                                         mlir::Type resultTy) {
  // Let the operand type be T1 C1::*, let the result type be T2 C2::*.
  // Verify that T1 and T2 are the same type.
  mlir::Type inputMemberTy;
  mlir::Type resultMemberTy;
  if (mlir::isa<cir::DataMemberType>(src.getType())) {
    inputMemberTy =
        mlir::cast<cir::DataMemberType>(src.getType()).getMemberTy();
    resultMemberTy = mlir::cast<cir::DataMemberType>(resultTy).getMemberTy();
  }
  assert(!cir::MissingFeatures::memberFuncPtrCast());
  if (inputMemberTy != resultMemberTy)
    return op->emitOpError()
           << "member types of the operand and the result do not match";

  return mlir::success();
}

LogicalResult cir::BaseDataMemberOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

LogicalResult cir::DerivedDataMemberOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

//===----------------------------------------------------------------------===//
// BaseMethodOp & DerivedMethodOp
//===----------------------------------------------------------------------===//

LogicalResult cir::BaseMethodOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

LogicalResult cir::DerivedMethodOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

//===----------------------------------------------------------------------===//
// GetRuntimeMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::GetRuntimeMemberOp::verify() {
  cir::DataMemberType memberPtrTy = getMember().getType();

  if (getAddr().getType().getPointee() != memberPtrTy.getClassTy())
    return emitError() << "record type does not match the member pointer type";
  if (getType().getPointee() != memberPtrTy.getMemberTy())
    return emitError() << "result type does not match the member pointer type";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetMethodOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::GetMethodOp::verify() {
  cir::MethodType methodTy = getMethod().getType();

  // Assume objectTy is !cir.ptr<!T>
  cir::PointerType objectPtrTy = getObject().getType();
  mlir::Type objectTy = objectPtrTy.getPointee();

  if (methodTy.getClassTy() != objectTy)
    return emitError() << "method class type and object type do not match";

  // Assume methodFuncTy is !cir.func<!Ret (!Args)>
  auto calleeTy = mlir::cast<cir::FuncType>(getCallee().getType().getPointee());
  cir::FuncType methodFuncTy = methodTy.getMemberFuncTy();

  // We verify at here that calleeTy is !cir.func<!Ret (!cir.ptr<!void>, !Args)>
  // Note that the first parameter type of the callee is !cir.ptr<!void> instead
  // of !cir.ptr<!T> because the "this" pointer may be adjusted before calling
  // the callee.

  if (methodFuncTy.getReturnType() != calleeTy.getReturnType())
    return emitError()
           << "method return type and callee return type do not match";

  llvm::ArrayRef<mlir::Type> calleeArgsTy = calleeTy.getInputs();
  llvm::ArrayRef<mlir::Type> methodFuncArgsTy = methodFuncTy.getInputs();

  if (calleeArgsTy.empty())
    return emitError() << "callee parameter list lacks receiver object ptr";

  auto calleeThisArgPtrTy = mlir::dyn_cast<cir::PointerType>(calleeArgsTy[0]);
  if (!calleeThisArgPtrTy ||
      !mlir::isa<cir::VoidType>(calleeThisArgPtrTy.getPointee())) {
    return emitError()
           << "the first parameter of callee must be a void pointer";
  }

  if (calleeArgsTy.size() != methodFuncArgsTy.size())
    return emitError() << "callee and method parameter counts do not match";

  if (calleeArgsTy.size() > 1 &&
      calleeArgsTy.slice(1) != methodFuncArgsTy.slice(1))
    return emitError()
           << "callee parameters and method parameters do not match";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsClassMembers.cpp.inc"
