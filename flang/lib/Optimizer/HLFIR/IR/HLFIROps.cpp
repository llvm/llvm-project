//===-- HLFIROps.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <tuple>

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

/// Given a FIR memory type, and information about non default lower bounds, get
/// the related HLFIR variable type.
mlir::Type hlfir::DeclareOp::getHLFIRVariableType(mlir::Type inputType,
                                                  bool hasExplicitLowerBounds) {
  mlir::Type type = fir::unwrapRefType(inputType);
  if (type.isa<fir::BaseBoxType>())
    return inputType;
  if (auto charType = type.dyn_cast<fir::CharacterType>())
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());

  auto seqType = type.dyn_cast<fir::SequenceType>();
  bool hasDynamicExtents =
      seqType && fir::sequenceWithNonConstantShape(seqType);
  mlir::Type eleType = seqType ? seqType.getEleTy() : type;
  bool hasDynamicLengthParams = fir::characterWithDynamicLen(eleType) ||
                                fir::isRecordWithTypeParameters(eleType);
  if (hasExplicitLowerBounds || hasDynamicExtents || hasDynamicLengthParams)
    return fir::BoxType::get(type);
  return inputType;
}

static bool hasExplicitLowerBounds(mlir::Value shape) {
  return shape && shape.getType().isa<fir::ShapeShiftType, fir::ShiftType>();
}

void hlfir::DeclareOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value memref,
                             llvm::StringRef uniq_name, mlir::Value shape,
                             mlir::ValueRange typeparams,
                             fir::FortranVariableFlagsAttr fortran_attrs) {
  auto nameAttr = builder.getStringAttr(uniq_name);
  mlir::Type inputType = memref.getType();
  bool hasExplicitLbs = hasExplicitLowerBounds(shape);
  mlir::Type hlfirVariableType =
      getHLFIRVariableType(inputType, hasExplicitLbs);
  build(builder, result, {hlfirVariableType, inputType}, memref, shape,
        typeparams, nameAttr, fortran_attrs);
}

mlir::LogicalResult hlfir::DeclareOp::verify() {
  if (getMemref().getType() != getResult(1).getType())
    return emitOpError("second result type must match input memref type");
  mlir::Type hlfirVariableType = getHLFIRVariableType(
      getMemref().getType(), hasExplicitLowerBounds(getShape()));
  if (hlfirVariableType != getResult(0).getType())
    return emitOpError("first result type is inconsistent with variable "
                       "properties: expected ")
           << hlfirVariableType;
  // The rest of the argument verification is done by the
  // FortranVariableInterface verifier.
  auto fortranVar =
      mlir::cast<fir::FortranVariableOpInterface>(this->getOperation());
  return fortranVar.verifyDeclareLikeOpImpl(getMemref());
}

#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"
