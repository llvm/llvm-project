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
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

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
                                hlfir::Entity entity) {
  if (auto variable = entity.getIfVariableInterface())
    return {hlfir::translateToExtendedValue(variable), {}};
  if (entity.isVariable())
    TODO(loc, "HLFIR variable to fir::ExtendedValue without a "
              "FortranVariableOpInterface");
  if (entity.getType().isa<hlfir::ExprType>())
    TODO(loc, "hlfir.expr to fir::ExtendedValue"); // use hlfir.associate
  return {{static_cast<mlir::Value>(entity)}, {}};
}

mlir::Value hlfir::Entity::getFirBase() const {
  if (fir::FortranVariableOpInterface variable = getIfVariableInterface())
    if (auto declareOp =
            mlir::dyn_cast<hlfir::DeclareOp>(variable.getOperation()))
      return declareOp.getOriginalBase();
  return getBase();
}

fir::ExtendedValue
hlfir::translateToExtendedValue(fir::FortranVariableOpInterface variable) {
  /// When going towards FIR, use the original base value to avoid
  /// introducing descriptors at runtime when they are not required.
  mlir::Value firBase = Entity{variable}.getFirBase();
  if (variable.isPointer() || variable.isAllocatable())
    TODO(variable->getLoc(), "pointer or allocatable "
                             "FortranVariableOpInterface to extendedValue");
  if (firBase.getType().isa<fir::BaseBoxType>())
    return fir::BoxValue(firBase, getExplicitLbounds(variable),
                         getExplicitTypeParams(variable));

  if (variable.isCharacter()) {
    if (variable.isArray())
      return fir::CharArrayBoxValue(firBase, variable.getExplicitCharLen(),
                                    getExplicitExtents(variable),
                                    getExplicitLbounds(variable));
    return fir::CharBoxValue(firBase, variable.getExplicitCharLen());
  }
  if (variable.isArray())
    return fir::ArrayBoxValue(firBase, getExplicitExtents(variable),
                              getExplicitLbounds(variable));
  return firBase;
}

hlfir::EntityWithAttributes
hlfir::genDeclare(mlir::Location loc, fir::FirOpBuilder &builder,
                  const fir::ExtendedValue &exv, llvm::StringRef name,
                  fir::FortranVariableFlagsAttr flags) {

  mlir::Value base = fir::getBase(exv);
  assert(fir::isa_passbyref_type(base.getType()) &&
         "entity being declared must be in memory");
  mlir::Value shapeOrShift;
  llvm::SmallVector<mlir::Value> lenParams;
  exv.match(
      [&](const fir::CharBoxValue &box) {
        lenParams.emplace_back(box.getLen());
      },
      [&](const fir::ArrayBoxValue &) {
        shapeOrShift = builder.createShape(loc, exv);
      },
      [&](const fir::CharArrayBoxValue &box) {
        shapeOrShift = builder.createShape(loc, exv);
        lenParams.emplace_back(box.getLen());
      },
      [&](const fir::BoxValue &box) {
        if (!box.getLBounds().empty())
          shapeOrShift = builder.createShape(loc, exv);
        lenParams.append(box.getExplicitParameters().begin(),
                         box.getExplicitParameters().end());
      },
      [&](const fir::MutableBoxValue &box) {
        lenParams.append(box.nonDeferredLenParams().begin(),
                         box.nonDeferredLenParams().end());
      },
      [](const auto &) {});
  auto declareOp = builder.create<hlfir::DeclareOp>(
      loc, base, name, shapeOrShift, lenParams, flags);
  return mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation());
}

/// If the entity is a variable, load its value (dereference pointers and
/// allocatables if needed). Do nothing if the entity os already a variable or
/// if it is not a scalar entity of numerical or logical type.
hlfir::Entity hlfir::loadTrivialScalar(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       Entity entity) {
  if (entity.isVariable() && entity.isScalar() &&
      fir::isa_trivial(entity.getFortranElementType())) {
    if (entity.isMutableBox())
      TODO(loc, "load pointer/allocatable scalar");
    return Entity{builder.create<fir::LoadOp>(loc, entity)};
  }
  return entity;
}
