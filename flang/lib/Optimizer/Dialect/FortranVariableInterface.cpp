//===-- FortranVariableInterface.cpp.cpp ----------------------------------===//
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

#include "flang/Optimizer/Dialect/FortranVariableInterface.h"

#include "flang/Optimizer/Dialect/FortranVariableInterface.cpp.inc"

llvm::LogicalResult
fir::FortranVariableOpInterface::verifyDeclareLikeOpImpl(mlir::Value memref) {
  const unsigned numExplicitTypeParams = getExplicitTypeParams().size();
  mlir::Type memType = memref.getType();
  const bool sourceIsBoxValue = mlir::isa<fir::BaseBoxType>(memType);
  const bool sourceIsBoxAddress = fir::isBoxAddress(memType);
  const bool sourceIsBox = sourceIsBoxValue || sourceIsBoxAddress;
  if (isCharacter()) {
    if (numExplicitTypeParams > 1)
      return emitOpError(
          "of character entity must have at most one length parameter");
    if (numExplicitTypeParams == 0 && !sourceIsBox)
      return emitOpError("must be provided exactly one type parameter when its "
                         "base is a character that is not a box");

  } else if (auto recordType =
                 mlir::dyn_cast<fir::RecordType>(getElementType())) {
    if (numExplicitTypeParams < recordType.getNumLenParams() && !sourceIsBox)
      return emitOpError("must be provided all the derived type length "
                         "parameters when the base is not a box");
    if (numExplicitTypeParams > recordType.getNumLenParams())
      return emitOpError("has too many length parameters");
  } else if (numExplicitTypeParams != 0) {
    return emitOpError("of numeric, logical, or assumed type entity must not "
                       "have length parameters");
  }

  if (isArray()) {
    if (mlir::Value shape = getShape()) {
      if (sourceIsBoxAddress)
        return emitOpError("for box address must not have a shape operand");
      unsigned shapeRank = 0;
      if (auto shapeType = mlir::dyn_cast<fir::ShapeType>(shape.getType())) {
        shapeRank = shapeType.getRank();
      } else if (auto shapeShiftType =
                     mlir::dyn_cast<fir::ShapeShiftType>(shape.getType())) {
        shapeRank = shapeShiftType.getRank();
      } else {
        if (!sourceIsBoxValue)
          emitOpError("of array entity with a raw address base must have a "
                      "shape operand that is a shape or shapeshift");
        shapeRank = mlir::cast<fir::ShiftType>(shape.getType()).getRank();
      }

      std::optional<unsigned> rank = getRank();
      if (!rank || *rank != shapeRank)
        return emitOpError("has conflicting shape and base operand ranks");
    } else if (!sourceIsBox) {
      emitOpError("of array entity with a raw address base must have a shape "
                  "operand that is a shape or shapeshift");
    }
  }
  return mlir::success();
}
