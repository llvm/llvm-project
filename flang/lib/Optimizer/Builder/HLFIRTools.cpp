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
#include "mlir/IR/BlockAndValueMapping.h"
#include <optional>

// Return explicit extents. If the base is a fir.box, this won't read it to
// return the extents and will instead return an empty vector.
static llvm::SmallVector<mlir::Value> getExplicitExtents(mlir::Value shape) {
  llvm::SmallVector<mlir::Value> result;
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
    TODO(shape.getLoc(), "read fir.shape to get extents");
  }
  return result;
}
static llvm::SmallVector<mlir::Value>
getExplicitExtents(fir::FortranVariableOpInterface var) {
  if (mlir::Value shape = var.getShape())
    return getExplicitExtents(var.getShape());
  return {};
}

// Return explicit lower bounds. For pointers and allocatables, this will not
// read the lower bounds and instead return an empty vector.
static llvm::SmallVector<mlir::Value> getExplicitLbounds(mlir::Value shape) {
  llvm::SmallVector<mlir::Value> result;
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
    TODO(shape.getLoc(), "read fir.shape to get lower bounds");
  }
  return result;
}
static llvm::SmallVector<mlir::Value>
getExplicitLbounds(fir::FortranVariableOpInterface var) {
  if (mlir::Value shape = var.getShape())
    return getExplicitLbounds(shape);
  return {};
}

static llvm::SmallVector<mlir::Value>
getExplicitTypeParams(fir::FortranVariableOpInterface var) {
  llvm::SmallVector<mlir::Value> res;
  mlir::OperandRange range = var.getExplicitTypeParams();
  res.append(range.begin(), range.end());
  return res;
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                                hlfir::Entity entity) {
  if (auto variable = entity.getIfVariableInterface())
    return {hlfir::translateToExtendedValue(loc, builder, variable), {}};
  if (entity.isVariable()) {
    if (entity.isScalar() && !entity.hasLengthParameters() &&
        !hlfir::isBoxAddressOrValueType(entity.getType()))
      return {fir::ExtendedValue{entity.getBase()}, std::nullopt};
    TODO(loc, "HLFIR variable to fir::ExtendedValue without a "
              "FortranVariableOpInterface");
  }
  if (entity.getType().isa<hlfir::ExprType>()) {
    hlfir::AssociateOp associate = hlfir::genAssociateExpr(
        loc, builder, entity, entity.getType(), "adapt.valuebyref");
    auto *bldr = &builder;
    hlfir::CleanupFunction cleanup = [bldr, loc, associate]() -> void {
      bldr->create<hlfir::EndAssociateOp>(loc, associate);
    };
    hlfir::Entity temp{associate.getBase()};
    return {translateToExtendedValue(loc, builder, temp).first, cleanup};
  }
  return {{static_cast<mlir::Value>(entity)}, {}};
}

mlir::Value hlfir::Entity::getFirBase() const {
  if (fir::FortranVariableOpInterface variable = getIfVariableInterface()) {
    if (auto declareOp =
            mlir::dyn_cast<hlfir::DeclareOp>(variable.getOperation()))
      return declareOp.getOriginalBase();
    if (auto associateOp =
            mlir::dyn_cast<hlfir::AssociateOp>(variable.getOperation()))
      return associateOp.getFirBase();
  }
  return getBase();
}

fir::ExtendedValue
hlfir::translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                                fir::FortranVariableOpInterface variable) {
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
    if (auto boxCharType = firBase.getType().dyn_cast<fir::BoxCharType>()) {
      auto unboxed = builder.create<fir::UnboxCharOp>(
          loc, fir::ReferenceType::get(boxCharType.getEleTy()),
          builder.getIndexType(), firBase);
      return fir::CharBoxValue(unboxed.getResult(0),
                               variable.getExplicitCharLen());
    }
    return fir::CharBoxValue(firBase, variable.getExplicitCharLen());
  }
  if (variable.isArray())
    return fir::ArrayBoxValue(firBase, getExplicitExtents(variable),
                              getExplicitLbounds(variable));
  return firBase;
}

fir::FortranVariableOpInterface
hlfir::genDeclare(mlir::Location loc, fir::FirOpBuilder &builder,
                  const fir::ExtendedValue &exv, llvm::StringRef name,
                  fir::FortranVariableFlagsAttr flags) {

  mlir::Value base = fir::getBase(exv);
  assert(fir::conformsWithPassByRef(base.getType()) &&
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

hlfir::AssociateOp hlfir::genAssociateExpr(mlir::Location loc,
                                           fir::FirOpBuilder &builder,
                                           hlfir::Entity value,
                                           mlir::Type variableType,
                                           llvm::StringRef name) {
  assert(value.isValue() && "must not be a variable");
  mlir::Value shape{};
  if (value.isArray())
    shape = genShape(loc, builder, value);

  mlir::Value source = value;
  // Lowered scalar expression values for numerical and logical may have a
  // different type than what is required for the type in memory (logical
  // expressions are typically manipulated as i1, but needs to be stored
  // according to the fir.logical<kind> so that the storage size is correct).
  // Character length mismatches are ignored (it is ok for one to be dynamic
  // and the other static).
  mlir::Type varEleTy = getFortranElementType(variableType);
  mlir::Type valueEleTy = getFortranElementType(value.getType());
  if (varEleTy != valueEleTy && !(valueEleTy.isa<fir::CharacterType>() &&
                                  varEleTy.isa<fir::CharacterType>())) {
    assert(value.isScalar() && fir::isa_trivial(value.getType()));
    source = builder.createConvert(loc, fir::unwrapPassByRefType(variableType),
                                   value);
  }
  llvm::SmallVector<mlir::Value> lenParams;
  genLengthParameters(loc, builder, value, lenParams);
  return builder.create<hlfir::AssociateOp>(loc, source, name, shape, lenParams,
                                            fir::FortranVariableFlagsAttr{});
}

mlir::Value hlfir::genVariableRawAddress(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity var) {
  assert(var.isVariable() && "only address of variables can be taken");
  mlir::Value baseAddr = var.getFirBase();
  if (var.isMutableBox())
    baseAddr = builder.create<fir::LoadOp>(loc, baseAddr);
  // Get raw address.
  if (baseAddr.getType().isa<fir::BaseBoxType>()) {
    auto addrType =
        fir::ReferenceType::get(fir::unwrapPassByRefType(baseAddr.getType()));
    baseAddr = builder.create<fir::BoxAddrOp>(loc, addrType, baseAddr);
  }
  return baseAddr;
}

mlir::Value hlfir::genVariableBoxChar(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::Entity var) {
  assert(var.isVariable() && "only address of variables can be taken");
  if (var.getType().isa<fir::BoxCharType>())
    return var;
  mlir::Value addr = genVariableRawAddress(loc, builder, var);
  llvm::SmallVector<mlir::Value> lengths;
  genLengthParameters(loc, builder, var, lengths);
  assert(lengths.size() == 1);
  auto charType = var.getFortranElementType().cast<fir::CharacterType>();
  auto boxCharType =
      fir::BoxCharType::get(builder.getContext(), charType.getFKind());
  auto scalarAddr =
      builder.createConvert(loc, fir::ReferenceType::get(charType), addr);
  return builder.create<fir::EmboxCharOp>(loc, boxCharType, scalarAddr,
                                          lengths[0]);
}

hlfir::Entity hlfir::loadTrivialScalar(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       Entity entity) {
  entity = derefPointersAndAllocatables(loc, builder, entity);
  if (entity.isVariable() && entity.isScalar() &&
      fir::isa_trivial(entity.getFortranElementType())) {
    return Entity{builder.create<fir::LoadOp>(loc, entity)};
  }
  return entity;
}

static std::optional<llvm::SmallVector<mlir::Value>>
getNonDefaultLowerBounds(mlir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::Entity entity) {
  if (!entity.hasNonDefaultLowerBounds())
    return std::nullopt;
  if (auto varIface = entity.getIfVariableInterface()) {
    llvm::SmallVector<mlir::Value> lbounds = getExplicitLbounds(varIface);
    if (!lbounds.empty())
      return lbounds;
  }
  TODO(loc, "get non default lower bounds without FortranVariableInterface");
}

hlfir::Entity hlfir::getElementAt(mlir::Location loc,
                                  fir::FirOpBuilder &builder, Entity entity,
                                  mlir::ValueRange oneBasedIndices) {
  if (entity.isScalar())
    return entity;
  llvm::SmallVector<mlir::Value> lenParams;
  genLengthParameters(loc, builder, entity, lenParams);
  if (entity.getType().isa<hlfir::ExprType>())
    return hlfir::Entity{builder.create<hlfir::ApplyOp>(
        loc, entity, oneBasedIndices, lenParams)};
  // Build hlfir.designate. The lower bounds may need to be added to
  // the oneBasedIndices since hlfir.designate expect indices
  // based on the array operand lower bounds.
  mlir::Type resultType = hlfir::getVariableElementType(entity);
  hlfir::DesignateOp designate;
  if (auto lbounds = getNonDefaultLowerBounds(loc, builder, entity)) {
    llvm::SmallVector<mlir::Value> indices;
    mlir::Type idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto [oneBased, lb] : llvm::zip(oneBasedIndices, *lbounds)) {
      auto lbIdx = builder.createConvert(loc, idxTy, lb);
      auto oneBasedIdx = builder.createConvert(loc, idxTy, oneBased);
      auto shift = builder.create<mlir::arith::SubIOp>(loc, lbIdx, one);
      mlir::Value index =
          builder.create<mlir::arith::AddIOp>(loc, oneBasedIdx, shift);
      indices.push_back(index);
    }
    designate = builder.create<hlfir::DesignateOp>(loc, resultType, entity,
                                                   indices, lenParams);
  } else {
    designate = builder.create<hlfir::DesignateOp>(loc, resultType, entity,
                                                   oneBasedIndices, lenParams);
  }
  return mlir::cast<fir::FortranVariableOpInterface>(designate.getOperation());
}

static mlir::Value genUBound(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lb, mlir::Value extent,
                             mlir::Value one) {
  if (auto constantLb = fir::getIntIfConstant(lb))
    if (*constantLb == 1)
      return extent;
  extent = builder.createConvert(loc, one.getType(), extent);
  lb = builder.createConvert(loc, one.getType(), lb);
  auto add = builder.create<mlir::arith::AddIOp>(loc, lb, extent);
  return builder.create<mlir::arith::SubIOp>(loc, add, one);
}

llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>
hlfir::genBounds(mlir::Location loc, fir::FirOpBuilder &builder,
                 Entity entity) {
  if (entity.getType().isa<hlfir::ExprType>())
    TODO(loc, "bounds of expressions in hlfir");
  auto [exv, cleanup] = translateToExtendedValue(loc, builder, entity);
  assert(!cleanup && "translation of entity should not yield cleanup");
  if (const auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>())
    exv = fir::factory::genMutableBoxRead(builder, loc, *mutableBox);
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> result;
  for (unsigned dim = 0; dim < exv.rank(); ++dim) {
    mlir::Value extent = fir::factory::readExtent(builder, loc, exv, dim);
    mlir::Value lb = fir::factory::readLowerBound(builder, loc, exv, dim, one);
    mlir::Value ub = genUBound(loc, builder, lb, extent, one);
    result.push_back({lb, ub});
  }
  return result;
}

llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>
hlfir::genBounds(mlir::Location loc, fir::FirOpBuilder &builder,
                 mlir::Value shape) {
  assert((shape.getType().isa<fir::ShapeShiftType>() ||
          shape.getType().isa<fir::ShapeType>()) &&
         "shape must contain extents");
  auto extents = getExplicitExtents(shape);
  auto lowers = getExplicitLbounds(shape);
  assert(lowers.empty() || lowers.size() == extents.size());
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> result;
  for (auto extent : llvm::enumerate(extents)) {
    mlir::Value lb = lowers.empty() ? one : lowers[extent.index()];
    mlir::Value ub = lowers.empty()
                         ? extent.value()
                         : genUBound(loc, builder, lb, extent.value(), one);
    result.push_back({lb, ub});
  }
  return result;
}

static hlfir::Entity followEntitySource(hlfir::Entity entity) {
  while (true) {
    if (auto reassoc = entity.getDefiningOp<hlfir::NoReassocOp>()) {
      entity = hlfir::Entity{reassoc.getVal()};
      continue;
    }
    if (auto asExpr = entity.getDefiningOp<hlfir::AsExprOp>()) {
      entity = hlfir::Entity{asExpr.getVar()};
      continue;
    }
    break;
  }
  return entity;
}

mlir::Value hlfir::genShape(mlir::Location loc, fir::FirOpBuilder &builder,
                            hlfir::Entity entity) {
  assert(entity.isArray() && "entity must be an array");
  if (entity.isMutableBox())
    entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
  else
    entity = followEntitySource(entity);

  if (entity.getType().isa<hlfir::ExprType>()) {
    if (auto elemental = entity.getDefiningOp<hlfir::ElementalOp>())
      return elemental.getShape();
    TODO(loc, "get shape from HLFIR expr without producer holding the shape");
  }
  // Entity is an array variable.
  if (auto varIface = entity.getIfVariableInterface()) {
    if (auto shape = varIface.getShape()) {
      if (shape.getType().isa<fir::ShapeType>())
        return shape;
      if (shape.getType().isa<fir::ShapeShiftType>())
        if (auto s = shape.getDefiningOp<fir::ShapeShiftOp>())
          return builder.create<fir::ShapeOp>(loc, s.getExtents());
    }
  }
  // There is no shape lying around for this entity: build one using
  // the type shape information, and/or the fir.box/fir.class shape
  // information if any extents are not static.
  fir::SequenceType seqTy =
      hlfir::getFortranElementOrSequenceType(entity.getType())
          .cast<fir::SequenceType>();
  llvm::SmallVector<mlir::Value> extents;
  mlir::Type idxTy = builder.getIndexType();
  for (auto typeExtent : seqTy.getShape())
    if (typeExtent != fir::SequenceType::getUnknownExtent()) {
      extents.push_back(builder.createIntegerConstant(loc, idxTy, typeExtent));
    } else {
      assert(entity.getType().isa<fir::BaseBoxType>() &&
             "array variable with dynamic extent must be boxes");
      mlir::Value dim =
          builder.createIntegerConstant(loc, idxTy, extents.size());
      auto dimInfo =
          builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, entity, dim);
      extents.push_back(dimInfo.getExtent());
    }
  return builder.create<fir::ShapeOp>(loc, extents);
}

llvm::SmallVector<mlir::Value>
hlfir::getIndexExtents(mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::Value shape) {
  llvm::SmallVector<mlir::Value> extents;
  if (auto s = shape.getDefiningOp<fir::ShapeOp>()) {
    auto e = s.getExtents();
    extents.insert(extents.end(), e.begin(), e.end());
  } else if (auto s = shape.getDefiningOp<fir::ShapeShiftOp>()) {
    auto e = s.getExtents();
    extents.insert(extents.end(), e.begin(), e.end());
  } else {
    // TODO: add fir.get_extent ops on fir.shape<> ops.
    TODO(loc, "get extents from fir.shape without fir::ShapeOp parent op");
  }
  mlir::Type indexType = builder.getIndexType();
  for (auto &extent : extents)
    extent = builder.createConvert(loc, indexType, extent);
  return extents;
}

void hlfir::genLengthParameters(mlir::Location loc, fir::FirOpBuilder &builder,
                                Entity entity,
                                llvm::SmallVectorImpl<mlir::Value> &result) {
  if (!entity.hasLengthParameters())
    return;
  if (entity.getType().isa<hlfir::ExprType>()) {
    mlir::Value expr = entity;
    if (auto reassoc = expr.getDefiningOp<hlfir::NoReassocOp>())
      expr = reassoc.getVal();
    // Going through fir::ExtendedValue would create a temp,
    // which is not desired for an inquiry.
    // TODO: make this an interface when adding further character producing ops.
    if (auto concat = expr.getDefiningOp<hlfir::ConcatOp>()) {
      result.push_back(concat.getLength());
      return;
    } else if (auto concat = expr.getDefiningOp<hlfir::SetLengthOp>()) {
      result.push_back(concat.getLength());
      return;
    } else if (auto asExpr = expr.getDefiningOp<hlfir::AsExprOp>()) {
      hlfir::genLengthParameters(loc, builder, hlfir::Entity{asExpr.getVar()},
                                 result);
      return;
    } else if (auto elemental = expr.getDefiningOp<hlfir::ElementalOp>()) {
      result.append(elemental.getTypeparams().begin(),
                    elemental.getTypeparams().end());
      return;
    } else if (auto apply = expr.getDefiningOp<hlfir::ApplyOp>()) {
      result.append(apply.getTypeparams().begin(), apply.getTypeparams().end());
    }
    TODO(loc, "inquire type parameters of hlfir.expr");
  }

  if (entity.isCharacter()) {
    auto [exv, cleanup] = translateToExtendedValue(loc, builder, entity);
    assert(!cleanup && "translation of entity should not yield cleanup");
    result.push_back(fir::factory::readCharLen(builder, loc, exv));
    return;
  }
  TODO(loc, "inquire PDTs length parameters in HLFIR");
}

mlir::Value hlfir::genCharLength(mlir::Location loc, fir::FirOpBuilder &builder,
                                 hlfir::Entity entity) {
  llvm::SmallVector<mlir::Value, 1> lenParams;
  genLengthParameters(loc, builder, entity, lenParams);
  assert(lenParams.size() == 1 && "characters must have one length parameters");
  return lenParams[0];
}

std::pair<mlir::Value, mlir::Value> hlfir::genVariableFirBaseShapeAndParams(
    mlir::Location loc, fir::FirOpBuilder &builder, Entity entity,
    llvm::SmallVectorImpl<mlir::Value> &typeParams) {
  auto [exv, cleanup] = translateToExtendedValue(loc, builder, entity);
  assert(!cleanup && "variable to Exv should not produce cleanup");
  if (entity.hasLengthParameters()) {
    auto params = fir::getTypeParams(exv);
    typeParams.append(params.begin(), params.end());
  }
  if (entity.isScalar())
    return {fir::getBase(exv), mlir::Value{}};
  if (auto variableInterface = entity.getIfVariableInterface())
    return {fir::getBase(exv), variableInterface.getShape()};
  return {fir::getBase(exv), builder.createShape(loc, exv)};
}

hlfir::Entity hlfir::derefPointersAndAllocatables(mlir::Location loc,
                                                  fir::FirOpBuilder &builder,
                                                  Entity entity) {
  if (entity.isMutableBox())
    return hlfir::Entity{builder.create<fir::LoadOp>(loc, entity).getResult()};
  return entity;
}

mlir::Type hlfir::getVariableElementType(hlfir::Entity variable) {
  assert(variable.isVariable() && "entity must be a variable");
  if (variable.isScalar())
    return variable.getType();
  mlir::Type eleTy = variable.getFortranElementType();
  if (variable.isPolymorphic())
    return fir::ClassType::get(eleTy);
  if (auto charType = eleTy.dyn_cast<fir::CharacterType>()) {
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());
  } else if (fir::isRecordWithTypeParameters(eleTy)) {
    return fir::BoxType::get(eleTy);
  }
  return fir::ReferenceType::get(eleTy);
}

static hlfir::ExprType getArrayExprType(mlir::Type elementType,
                                        mlir::Value shape, bool isPolymorphic) {
  unsigned rank = shape.getType().cast<fir::ShapeType>().getRank();
  hlfir::ExprType::Shape typeShape(rank, hlfir::ExprType::getUnknownExtent());
  if (auto shapeOp = shape.getDefiningOp<fir::ShapeOp>())
    for (auto extent : llvm::enumerate(shapeOp.getExtents()))
      if (auto cstExtent = fir::getIntIfConstant(extent.value()))
        typeShape[extent.index()] = *cstExtent;
  return hlfir::ExprType::get(elementType.getContext(), typeShape, elementType,
                              isPolymorphic);
}

hlfir::ElementalOp
hlfir::genElementalOp(mlir::Location loc, fir::FirOpBuilder &builder,
                      mlir::Type elementType, mlir::Value shape,
                      mlir::ValueRange typeParams,
                      const ElementalKernelGenerator &genKernel) {
  mlir::Type exprType = getArrayExprType(elementType, shape, false);
  auto elementalOp =
      builder.create<hlfir::ElementalOp>(loc, exprType, shape, typeParams);
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(elementalOp.getBody());
  mlir::Value elementResult = genKernel(loc, builder, elementalOp.getIndices());
  // Numerical and logical scalars may be lowered to another type than the
  // Fortran expression type (e.g i1 instead of fir.logical). Array expression
  // values are typed according to their Fortran type. Insert a cast if needed
  // here.
  if (fir::isa_trivial(elementResult.getType()))
    elementResult = builder.createConvert(loc, elementType, elementResult);
  builder.create<hlfir::YieldElementOp>(loc, elementResult);
  builder.restoreInsertionPoint(insertPt);
  return elementalOp;
}

hlfir::YieldElementOp
hlfir::inlineElementalOp(mlir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::ElementalOp elemental,
                         mlir::ValueRange oneBasedIndices) {
  // hlfir.elemental region is a SizedRegion<1>.
  assert(elemental.getRegion().hasOneBlock() &&
         "expect elemental region to have one block");
  mlir::BlockAndValueMapping mapper;
  mapper.map(elemental.getIndices(), oneBasedIndices);
  mlir::Operation *newOp;
  for (auto &op : elemental.getRegion().back().getOperations())
    newOp = builder.clone(op, mapper);
  auto yield = mlir::dyn_cast_or_null<hlfir::YieldElementOp>(newOp);
  assert(yield && "last ElementalOp operation must be am hlfir.yield_element");
  return yield;
}

std::pair<fir::DoLoopOp, llvm::SmallVector<mlir::Value>>
hlfir::genLoopNest(mlir::Location loc, fir::FirOpBuilder &builder,
                   mlir::ValueRange extents) {
  assert(!extents.empty() && "must have at least one extent");
  auto insPt = builder.saveInsertionPoint();
  llvm::SmallVector<mlir::Value> indices(extents.size());
  // Build loop nest from column to row.
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Type indexType = builder.getIndexType();
  unsigned dim = extents.size() - 1;
  fir::DoLoopOp innerLoop;
  for (auto extent : llvm::reverse(extents)) {
    auto ub = builder.createConvert(loc, indexType, extent);
    innerLoop = builder.create<fir::DoLoopOp>(loc, one, ub, one);
    builder.setInsertionPointToStart(innerLoop.getBody());
    // Reverse the indices so they are in column-major order.
    indices[dim--] = innerLoop.getInductionVar();
  }
  builder.restoreInsertionPoint(insPt);
  return {innerLoop, indices};
}
