//===-- HLFIRTools.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools to manipulate HLFIR variable and expressions
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"

// Return explicit extents. If the base is a fir.box, this won't read it to
// return the extents and will instead return an empty vector.
static llvm::SmallVector<mlir::Value>
getExplicitExtents(fir::FortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> result;
  if (mlir::Value shape = var.getShape()) {
    auto *shapeOp = shape.getDefiningOp();
    if (auto s = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
      auto e = s.getExtents();
      result.append(e.begin(), e.end());
    } else if (auto s = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeOp)) {
      auto e = s.getExtents();
      result.append(e.begin(), e.end());
    } else if (mlir::dyn_cast_or_null<fir::ShiftOp>(shapeOp)) {
      return {};
    } else {
      TODO(var->getLoc(), "read fir.shape to get extents");
    }
  }
  return result;
}

// Return explicit lower bounds. For pointers and allocatables, this will not
// read the lower bounds and instead return an empty vector.
static llvm::SmallVector<mlir::Value>
getExplicitLbounds(fir::FortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> result;
  if (mlir::Value shape = var.getShape()) {
    auto *shapeOp = shape.getDefiningOp();
    if (auto s = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
      return {};
    } else if (auto s = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeOp)) {
      auto e = s.getOrigins();
      result.append(e.begin(), e.end());
    } else if (auto s = mlir::dyn_cast_or_null<fir::ShiftOp>(shapeOp)) {
      auto e = s.getOrigins();
      result.append(e.begin(), e.end());
    } else {
      TODO(var->getLoc(), "read fir.shape to get lower bounds");
    }
  }
  return result;
}

static llvm::SmallVector<mlir::Value>
getExplicitTypeParams(fir::FortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> res;
  mlir::OperandRange range = var.getExplicitTypeParams();
  res.append(range.begin(), range.end());
  return res;
}

std::pair<fir::ExtendedValue, llvm::Optional<hlfir::CleanupFunction>>
hlfir::translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &,
                                hlfir::FortranEntity entity) {
  if (auto variable = entity.getIfVariable())
    return {hlfir::translateToExtendedValue(variable), {}};
  if (entity.getType().isa<hlfir::ExprType>())
    TODO(loc, "hlfir.expr to fir::ExtendedValue"); // use hlfir.associate
  return {{static_cast<mlir::Value>(entity)}, {}};
}

fir::ExtendedValue
hlfir::translateToExtendedValue(fir::FortranVariableOpInterface variable) {
  if (variable.isPointer() || variable.isAllocatable())
    TODO(variable->getLoc(), "pointer or allocatable "
                             "FortranVariableOpInterface to extendedValue");
  if (variable.getBase().getType().isa<fir::BaseBoxType>())
    return fir::BoxValue(variable.getBase(), getExplicitLbounds(variable),
                         getExplicitTypeParams(variable),
                         getExplicitExtents(variable));
  if (variable.isCharacter()) {
    if (variable.isArray())
      return fir::CharArrayBoxValue(
          variable.getBase(), variable.getExplicitCharLen(),
          getExplicitExtents(variable), getExplicitLbounds(variable));
    return fir::CharBoxValue(variable.getBase(), variable.getExplicitCharLen());
  }
  if (variable.isArray())
    return fir::ArrayBoxValue(variable.getBase(), getExplicitExtents(variable),
                              getExplicitLbounds(variable));
  return variable.getBase();
}
