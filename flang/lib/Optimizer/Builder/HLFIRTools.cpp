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
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <optional>

// Return explicit extents. If the base is a fir.box, this won't read it to
// return the extents and will instead return an empty vector.
llvm::SmallVector<mlir::Value>
hlfir::getExplicitExtentsFromShape(mlir::Value shape,
                                   fir::FirOpBuilder &builder) {
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
  } else if (auto s = mlir::dyn_cast_or_null<hlfir::ShapeOfOp>(shapeOp)) {
    hlfir::ExprType expr = mlir::cast<hlfir::ExprType>(s.getExpr().getType());
    llvm::ArrayRef<int64_t> exprShape = expr.getShape();
    mlir::Type indexTy = builder.getIndexType();
    fir::ShapeType shapeTy = mlir::cast<fir::ShapeType>(shape.getType());
    result.reserve(shapeTy.getRank());
    for (unsigned i = 0; i < shapeTy.getRank(); ++i) {
      int64_t extent = exprShape[i];
      mlir::Value extentVal;
      if (extent == expr.getUnknownExtent()) {
        auto op = hlfir::GetExtentOp::create(builder, shape.getLoc(), shape, i);
        extentVal = op.getResult();
      } else {
        extentVal =
            builder.createIntegerConstant(shape.getLoc(), indexTy, extent);
      }
      result.emplace_back(extentVal);
    }
  } else {
    TODO(shape.getLoc(), "read fir.shape to get extents");
  }
  return result;
}
static llvm::SmallVector<mlir::Value>
getExplicitExtents(fir::FortranVariableOpInterface var,
                   fir::FirOpBuilder &builder) {
  if (mlir::Value shape = var.getShape())
    return hlfir::getExplicitExtentsFromShape(var.getShape(), builder);
  return {};
}

// Return explicit lower bounds from a shape result.
// Only fir.shape, fir.shift and fir.shape_shift are currently
// supported as shape.
static llvm::SmallVector<mlir::Value>
getExplicitLboundsFromShape(mlir::Value shape) {
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

// Return explicit lower bounds. For pointers and allocatables, this will not
// read the lower bounds and instead return an empty vector.
static llvm::SmallVector<mlir::Value>
getExplicitLbounds(fir::FortranVariableOpInterface var) {
  if (mlir::Value shape = var.getShape())
    return getExplicitLboundsFromShape(shape);
  return {};
}

static llvm::SmallVector<mlir::Value>
getNonDefaultLowerBounds(mlir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::Entity entity) {
  assert(!entity.isAssumedRank() &&
         "cannot compute assumed rank bounds statically");
  if (!entity.mayHaveNonDefaultLowerBounds())
    return {};
  if (auto varIface = entity.getIfVariableInterface()) {
    llvm::SmallVector<mlir::Value> lbounds = getExplicitLbounds(varIface);
    if (!lbounds.empty())
      return lbounds;
  }
  if (entity.isMutableBox())
    entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
  llvm::SmallVector<mlir::Value> lowerBounds;
  fir::factory::genDimInfoFromBox(builder, loc, entity, &lowerBounds,
                                  /*extents=*/nullptr, /*strides=*/nullptr);
  return lowerBounds;
}

static llvm::SmallVector<mlir::Value> toSmallVector(mlir::ValueRange range) {
  llvm::SmallVector<mlir::Value> res;
  res.append(range.begin(), range.end());
  return res;
}

static llvm::SmallVector<mlir::Value> getExplicitTypeParams(hlfir::Entity var) {
  if (auto varIface = var.getMaybeDereferencedVariableInterface())
    return toSmallVector(varIface.getExplicitTypeParams());
  return {};
}

static mlir::Value tryGettingNonDeferredCharLen(hlfir::Entity var) {
  if (auto varIface = var.getMaybeDereferencedVariableInterface())
    if (!varIface.getExplicitTypeParams().empty())
      return varIface.getExplicitTypeParams()[0];
  return mlir::Value{};
}

static mlir::Value genCharacterVariableLength(mlir::Location loc,
                                              fir::FirOpBuilder &builder,
                                              hlfir::Entity var) {
  if (mlir::Value len = tryGettingNonDeferredCharLen(var))
    return len;
  auto charType = mlir::cast<fir::CharacterType>(var.getFortranElementType());
  if (charType.hasConstantLen())
    return builder.createIntegerConstant(loc, builder.getIndexType(),
                                         charType.getLen());
  if (var.isMutableBox())
    var = hlfir::Entity{fir::LoadOp::create(builder, loc, var)};
  mlir::Value len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
      var.getFirBase());
  assert(len && "failed to retrieve length");
  return len;
}

static fir::CharBoxValue genUnboxChar(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      mlir::Value boxChar) {
  if (auto emboxChar = boxChar.getDefiningOp<fir::EmboxCharOp>())
    return {emboxChar.getMemref(), emboxChar.getLen()};
  mlir::Type refType = fir::ReferenceType::get(
      mlir::cast<fir::BoxCharType>(boxChar.getType()).getEleTy());
  auto unboxed = fir::UnboxCharOp::create(builder, loc, refType,
                                          builder.getIndexType(), boxChar);
  mlir::Value addr = unboxed.getResult(0);
  mlir::Value len = unboxed.getResult(1);
  if (auto varIface = boxChar.getDefiningOp<fir::FortranVariableOpInterface>())
    if (mlir::Value explicitlen = varIface.getExplicitCharLen())
      len = explicitlen;
  return {addr, len};
}

// To maximize chances of identifying usage of a same variables in the IR,
// always return the hlfirBase result of declare/associate if it is a raw
// pointer.
static mlir::Value getFirBaseHelper(mlir::Value hlfirBase,
                                    mlir::Value firBase) {
  if (fir::isa_ref_type(hlfirBase.getType()))
    return hlfirBase;
  return firBase;
}

mlir::Value hlfir::Entity::getFirBase() const {
  if (fir::FortranVariableOpInterface variable = getIfVariableInterface()) {
    if (auto declareOp =
            mlir::dyn_cast<hlfir::DeclareOp>(variable.getOperation()))
      return getFirBaseHelper(declareOp.getBase(), declareOp.getOriginalBase());
    if (auto associateOp =
            mlir::dyn_cast<hlfir::AssociateOp>(variable.getOperation()))
      return getFirBaseHelper(associateOp.getBase(), associateOp.getFirBase());
  }
  return getBase();
}

static bool isShapeWithLowerBounds(mlir::Value shape) {
  if (!shape)
    return false;
  auto shapeTy = shape.getType();
  return mlir::isa<fir::ShiftType>(shapeTy) ||
         mlir::isa<fir::ShapeShiftType>(shapeTy);
}

bool hlfir::Entity::mayHaveNonDefaultLowerBounds() const {
  if (!isBoxAddressOrValue() || isScalar())
    return false;
  if (isMutableBox())
    return true;
  if (auto varIface = getIfVariableInterface())
    return isShapeWithLowerBounds(varIface.getShape());
  // Go through chain of fir.box converts.
  if (auto convert = getDefiningOp<fir::ConvertOp>()) {
    return hlfir::Entity{convert.getValue()}.mayHaveNonDefaultLowerBounds();
  } else if (auto rebox = getDefiningOp<fir::ReboxOp>()) {
    // If slicing is involved, then the resulting box has
    // default lower bounds. If there is no slicing,
    // then the result depends on the shape operand
    // (whether it has non default lower bounds or not).
    return !rebox.getSlice() && isShapeWithLowerBounds(rebox.getShape());
  } else if (auto embox = getDefiningOp<fir::EmboxOp>()) {
    return !embox.getSlice() && isShapeWithLowerBounds(embox.getShape());
  }
  return true;
}

mlir::Operation *traverseConverts(mlir::Operation *op) {
  while (auto convert = llvm::dyn_cast_or_null<fir::ConvertOp>(op))
    op = convert.getValue().getDefiningOp();
  return op;
}

bool hlfir::Entity::mayBeOptional() const {
  if (!isVariable())
    return false;
  // TODO: introduce a fir type to better identify optionals.
  if (mlir::Operation *op = traverseConverts(getDefiningOp())) {
    if (auto varIface = llvm::dyn_cast<fir::FortranVariableOpInterface>(op))
      return varIface.isOptional();
    return !llvm::isa<fir::AllocaOp, fir::AllocMemOp, fir::ReboxOp,
                      fir::EmboxOp, fir::LoadOp>(op);
  }
  return true;
}

fir::FortranVariableOpInterface
hlfir::genDeclare(mlir::Location loc, fir::FirOpBuilder &builder,
                  const fir::ExtendedValue &exv, llvm::StringRef name,
                  fir::FortranVariableFlagsAttr flags, mlir::Value dummyScope,
                  cuf::DataAttributeAttr dataAttr) {

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
  auto declareOp =
      hlfir::DeclareOp::create(builder, loc, base, name, shapeOrShift,
                               lenParams, dummyScope, flags, dataAttr);
  return mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation());
}

hlfir::AssociateOp
hlfir::genAssociateExpr(mlir::Location loc, fir::FirOpBuilder &builder,
                        hlfir::Entity value, mlir::Type variableType,
                        llvm::StringRef name,
                        std::optional<mlir::NamedAttribute> attr) {
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
  if (varEleTy != valueEleTy && !(mlir::isa<fir::CharacterType>(valueEleTy) &&
                                  mlir::isa<fir::CharacterType>(varEleTy))) {
    assert(value.isScalar() && fir::isa_trivial(value.getType()));
    source = builder.createConvert(loc, fir::unwrapPassByRefType(variableType),
                                   value);
  }
  llvm::SmallVector<mlir::Value> lenParams;
  genLengthParameters(loc, builder, value, lenParams);
  if (attr) {
    assert(name.empty() && "It attribute is provided, no-name is expected");
    return hlfir::AssociateOp::create(builder, loc, source, shape, lenParams,
                                      fir::FortranVariableFlagsAttr{},
                                      llvm::ArrayRef{*attr});
  }
  return hlfir::AssociateOp::create(builder, loc, source, name, shape,
                                    lenParams, fir::FortranVariableFlagsAttr{});
}

mlir::Value hlfir::genVariableRawAddress(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity var) {
  assert(var.isVariable() && "only address of variables can be taken");
  mlir::Value baseAddr = var.getFirBase();
  if (var.isMutableBox())
    baseAddr = fir::LoadOp::create(builder, loc, baseAddr);
  // Get raw address.
  if (mlir::isa<fir::BoxCharType>(var.getType()))
    baseAddr = genUnboxChar(loc, builder, var.getBase()).getAddr();
  if (mlir::isa<fir::BaseBoxType>(baseAddr.getType()))
    baseAddr = fir::BoxAddrOp::create(builder, loc, baseAddr);
  return baseAddr;
}

mlir::Value hlfir::genVariableBoxChar(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::Entity var) {
  assert(var.isVariable() && "only address of variables can be taken");
  if (mlir::isa<fir::BoxCharType>(var.getType()))
    return var;
  mlir::Value addr = genVariableRawAddress(loc, builder, var);
  llvm::SmallVector<mlir::Value> lengths;
  genLengthParameters(loc, builder, var, lengths);
  assert(lengths.size() == 1);
  auto charType = mlir::cast<fir::CharacterType>(var.getFortranElementType());
  auto boxCharType =
      fir::BoxCharType::get(builder.getContext(), charType.getFKind());
  auto scalarAddr =
      builder.createConvert(loc, fir::ReferenceType::get(charType), addr);
  return fir::EmboxCharOp::create(builder, loc, boxCharType, scalarAddr,
                                  lengths[0]);
}

static hlfir::Entity changeBoxAttributes(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity var,
                                         fir::BaseBoxType forceBoxType) {
  assert(llvm::isa<fir::BaseBoxType>(var.getType()) && "expect box type");
  // Propagate lower bounds.
  mlir::Value shift;
  llvm::SmallVector<mlir::Value> lbounds =
      getNonDefaultLowerBounds(loc, builder, var);
  if (!lbounds.empty())
    shift = builder.genShift(loc, lbounds);
  auto rebox = fir::ReboxOp::create(builder, loc, forceBoxType, var, shift,
                                    /*slice=*/nullptr);
  return hlfir::Entity{rebox};
}

hlfir::Entity hlfir::genVariableBox(mlir::Location loc,
                                    fir::FirOpBuilder &builder,
                                    hlfir::Entity var,
                                    fir::BaseBoxType forceBoxType) {
  assert(var.isVariable() && "must be a variable");
  var = hlfir::derefPointersAndAllocatables(loc, builder, var);
  if (mlir::isa<fir::BaseBoxType>(var.getType())) {
    if (!forceBoxType || forceBoxType == var.getType())
      return var;
    return changeBoxAttributes(loc, builder, var, forceBoxType);
  }
  // Note: if the var is not a fir.box/fir.class at that point, it has default
  // lower bounds and is not polymorphic.
  mlir::Value shape =
      var.isArray() ? hlfir::genShape(loc, builder, var) : mlir::Value{};
  llvm::SmallVector<mlir::Value> typeParams;
  mlir::Type elementType =
      forceBoxType ? fir::getFortranElementType(forceBoxType.getEleTy())
                   : var.getFortranElementType();
  auto maybeCharType = mlir::dyn_cast<fir::CharacterType>(elementType);
  if (!maybeCharType || maybeCharType.hasDynamicLen())
    hlfir::genLengthParameters(loc, builder, var, typeParams);
  mlir::Value addr = var.getBase();
  if (mlir::isa<fir::BoxCharType>(var.getType()))
    addr = genVariableRawAddress(loc, builder, var);
  const bool isVolatile = fir::isa_volatile_type(var.getType());
  mlir::Type boxType =
      fir::BoxType::get(var.getElementOrSequenceType(), isVolatile);
  if (forceBoxType) {
    boxType = forceBoxType;
    mlir::Type baseType =
        fir::ReferenceType::get(fir::unwrapRefType(forceBoxType.getEleTy()));
    addr = builder.createConvert(loc, baseType, addr);
  }
  auto embox = fir::EmboxOp::create(builder, loc, boxType, addr, shape,
                                    /*slice=*/mlir::Value{}, typeParams);
  return hlfir::Entity{embox.getResult()};
}

hlfir::Entity hlfir::loadTrivialScalar(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       Entity entity) {
  entity = derefPointersAndAllocatables(loc, builder, entity);
  if (entity.isVariable() && entity.isScalar() &&
      fir::isa_trivial(entity.getFortranElementType())) {
    // Optional entities may be represented with !fir.box<i32/f32/...>.
    // We need to take the data pointer before loading the scalar.
    mlir::Value base = genVariableRawAddress(loc, builder, entity);
    return Entity{fir::LoadOp::create(builder, loc, base)};
  }
  return entity;
}

hlfir::Entity hlfir::getElementAt(mlir::Location loc,
                                  fir::FirOpBuilder &builder, Entity entity,
                                  mlir::ValueRange oneBasedIndices) {
  if (entity.isScalar())
    return entity;
  llvm::SmallVector<mlir::Value> lenParams;
  genLengthParameters(loc, builder, entity, lenParams);
  if (mlir::isa<hlfir::ExprType>(entity.getType()))
    return hlfir::Entity{hlfir::ApplyOp::create(builder, loc, entity,
                                                oneBasedIndices, lenParams)};
  // Build hlfir.designate. The lower bounds may need to be added to
  // the oneBasedIndices since hlfir.designate expect indices
  // based on the array operand lower bounds.
  mlir::Type resultType = hlfir::getVariableElementType(entity);
  hlfir::DesignateOp designate;
  llvm::SmallVector<mlir::Value> lbounds =
      getNonDefaultLowerBounds(loc, builder, entity);
  if (!lbounds.empty()) {
    llvm::SmallVector<mlir::Value> indices;
    mlir::Type idxTy = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto [oneBased, lb] : llvm::zip(oneBasedIndices, lbounds)) {
      auto lbIdx = builder.createConvert(loc, idxTy, lb);
      auto oneBasedIdx = builder.createConvert(loc, idxTy, oneBased);
      auto shift = mlir::arith::SubIOp::create(builder, loc, lbIdx, one);
      mlir::Value index =
          mlir::arith::AddIOp::create(builder, loc, oneBasedIdx, shift);
      indices.push_back(index);
    }
    designate = hlfir::DesignateOp::create(builder, loc, resultType, entity,
                                           indices, lenParams);
  } else {
    designate = hlfir::DesignateOp::create(builder, loc, resultType, entity,
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
  auto add = mlir::arith::AddIOp::create(builder, loc, lb, extent);
  return mlir::arith::SubIOp::create(builder, loc, add, one);
}

llvm::SmallVector<std::pair<mlir::Value, mlir::Value>>
hlfir::genBounds(mlir::Location loc, fir::FirOpBuilder &builder,
                 Entity entity) {
  if (mlir::isa<hlfir::ExprType>(entity.getType()))
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
  assert((mlir::isa<fir::ShapeShiftType>(shape.getType()) ||
          mlir::isa<fir::ShapeType>(shape.getType())) &&
         "shape must contain extents");
  auto extents = hlfir::getExplicitExtentsFromShape(shape, builder);
  auto lowers = getExplicitLboundsFromShape(shape);
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

llvm::SmallVector<mlir::Value> hlfir::genLowerbounds(mlir::Location loc,
                                                     fir::FirOpBuilder &builder,
                                                     mlir::Value shape,
                                                     unsigned rank) {
  llvm::SmallVector<mlir::Value> lbounds;
  if (shape)
    lbounds = getExplicitLboundsFromShape(shape);
  if (!lbounds.empty())
    return lbounds;
  mlir::Value one =
      builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  return llvm::SmallVector<mlir::Value>(rank, one);
}

static hlfir::Entity followShapeInducingSource(hlfir::Entity entity) {
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

static mlir::Value computeVariableExtent(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity variable,
                                         fir::SequenceType seqTy,
                                         unsigned dim) {
  mlir::Type idxTy = builder.getIndexType();
  if (seqTy.getShape().size() > dim) {
    fir::SequenceType::Extent typeExtent = seqTy.getShape()[dim];
    if (typeExtent != fir::SequenceType::getUnknownExtent())
      return builder.createIntegerConstant(loc, idxTy, typeExtent);
  }
  assert(mlir::isa<fir::BaseBoxType>(variable.getType()) &&
         "array variable with dynamic extent must be boxed");
  mlir::Value dimVal = builder.createIntegerConstant(loc, idxTy, dim);
  auto dimInfo = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy,
                                        variable, dimVal);
  return dimInfo.getExtent();
}
llvm::SmallVector<mlir::Value> getVariableExtents(mlir::Location loc,
                                                  fir::FirOpBuilder &builder,
                                                  hlfir::Entity variable) {
  llvm::SmallVector<mlir::Value> extents;
  if (fir::FortranVariableOpInterface varIface =
          variable.getIfVariableInterface()) {
    extents = getExplicitExtents(varIface, builder);
    if (!extents.empty())
      return extents;
  }

  if (variable.isMutableBox())
    variable = hlfir::derefPointersAndAllocatables(loc, builder, variable);
  // Use the type shape information, and/or the fir.box/fir.class shape
  // information if any extents are not static.
  fir::SequenceType seqTy = mlir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(variable.getType()));
  unsigned rank = seqTy.getShape().size();
  for (unsigned dim = 0; dim < rank; ++dim)
    extents.push_back(
        computeVariableExtent(loc, builder, variable, seqTy, dim));
  return extents;
}

static mlir::Value tryRetrievingShapeOrShift(hlfir::Entity entity) {
  if (mlir::isa<hlfir::ExprType>(entity.getType())) {
    if (auto elemental = entity.getDefiningOp<hlfir::ElementalOp>())
      return elemental.getShape();
    if (auto evalInMem = entity.getDefiningOp<hlfir::EvaluateInMemoryOp>())
      return evalInMem.getShape();
    return mlir::Value{};
  }
  if (auto varIface = entity.getIfVariableInterface())
    return varIface.getShape();
  return {};
}

mlir::Value hlfir::genShape(mlir::Location loc, fir::FirOpBuilder &builder,
                            hlfir::Entity entity) {
  assert(entity.isArray() && "entity must be an array");
  entity = followShapeInducingSource(entity);
  assert(entity && "what?");
  if (auto shape = tryRetrievingShapeOrShift(entity)) {
    if (mlir::isa<fir::ShapeType>(shape.getType()))
      return shape;
    if (mlir::isa<fir::ShapeShiftType>(shape.getType()))
      if (auto s = shape.getDefiningOp<fir::ShapeShiftOp>())
        return fir::ShapeOp::create(builder, loc, s.getExtents());
  }
  if (mlir::isa<hlfir::ExprType>(entity.getType()))
    return hlfir::ShapeOfOp::create(builder, loc, entity.getBase());
  // There is no shape lying around for this entity. Retrieve the extents and
  // build a new fir.shape.
  return fir::ShapeOp::create(builder, loc,
                              getVariableExtents(loc, builder, entity));
}

llvm::SmallVector<mlir::Value>
hlfir::getIndexExtents(mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::Value shape) {
  llvm::SmallVector<mlir::Value> extents =
      hlfir::getExplicitExtentsFromShape(shape, builder);
  mlir::Type indexType = builder.getIndexType();
  for (auto &extent : extents)
    extent = builder.createConvert(loc, indexType, extent);
  return extents;
}

mlir::Value hlfir::genExtent(mlir::Location loc, fir::FirOpBuilder &builder,
                             hlfir::Entity entity, unsigned dim) {
  entity = followShapeInducingSource(entity);
  if (auto shape = tryRetrievingShapeOrShift(entity)) {
    auto extents = hlfir::getExplicitExtentsFromShape(shape, builder);
    if (!extents.empty()) {
      assert(extents.size() > dim && "bad inquiry");
      return extents[dim];
    }
  }
  if (entity.isVariable()) {
    if (entity.isMutableBox())
      entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
    // Use the type shape information, and/or the fir.box/fir.class shape
    // information if any extents are not static.
    fir::SequenceType seqTy = mlir::cast<fir::SequenceType>(
        hlfir::getFortranElementOrSequenceType(entity.getType()));
    return computeVariableExtent(loc, builder, entity, seqTy, dim);
  }
  TODO(loc, "get extent from HLFIR expr without producer holding the shape");
}

mlir::Value hlfir::genLBound(mlir::Location loc, fir::FirOpBuilder &builder,
                             hlfir::Entity entity, unsigned dim) {
  if (!entity.mayHaveNonDefaultLowerBounds())
    return builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  if (auto shape = tryRetrievingShapeOrShift(entity)) {
    auto lbounds = getExplicitLboundsFromShape(shape);
    if (!lbounds.empty()) {
      assert(lbounds.size() > dim && "bad inquiry");
      return lbounds[dim];
    }
  }
  if (entity.isMutableBox())
    entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
  assert(mlir::isa<fir::BaseBoxType>(entity.getType()) && "must be a box");
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value dimVal = builder.createIntegerConstant(loc, idxTy, dim);
  auto dimInfo =
      fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, entity, dimVal);
  return dimInfo.getLowerBound();
}

void hlfir::genLengthParameters(mlir::Location loc, fir::FirOpBuilder &builder,
                                Entity entity,
                                llvm::SmallVectorImpl<mlir::Value> &result) {
  if (!entity.hasLengthParameters())
    return;
  if (mlir::isa<hlfir::ExprType>(entity.getType())) {
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
    } else if (auto evalInMem =
                   expr.getDefiningOp<hlfir::EvaluateInMemoryOp>()) {
      result.append(evalInMem.getTypeparams().begin(),
                    evalInMem.getTypeparams().end());
      return;
    } else if (auto apply = expr.getDefiningOp<hlfir::ApplyOp>()) {
      result.append(apply.getTypeparams().begin(), apply.getTypeparams().end());
      return;
    }
    if (entity.isCharacter()) {
      result.push_back(hlfir::GetLengthOp::create(builder, loc, expr));
      return;
    }
    TODO(loc, "inquire PDTs length parameters of hlfir.expr");
  }

  if (entity.isCharacter()) {
    result.push_back(genCharacterVariableLength(loc, builder, entity));
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

mlir::Value hlfir::genRank(mlir::Location loc, fir::FirOpBuilder &builder,
                           hlfir::Entity entity, mlir::Type resultType) {
  if (!entity.isAssumedRank())
    return builder.createIntegerConstant(loc, resultType, entity.getRank());
  assert(entity.isBoxAddressOrValue() &&
         "assumed-ranks are box addresses or values");
  return fir::BoxRankOp::create(builder, loc, resultType, entity);
}

// Return a "shape" that can be used in fir.embox/fir.rebox with \p exv base.
static mlir::Value asEmboxShape(mlir::Location loc, fir::FirOpBuilder &builder,
                                const fir::ExtendedValue &exv,
                                mlir::Value shape) {
  if (!shape)
    return shape;
  // fir.rebox does not need and does not accept extents (fir.shape or
  // fir.shape_shift) since this information is already in the input fir.box,
  // it only accepts fir.shift because local lower bounds may not be reflected
  // in the fir.box.
  if (mlir::isa<fir::BaseBoxType>(fir::getBase(exv).getType()) &&
      !mlir::isa<fir::ShiftType>(shape.getType()))
    return builder.createShape(loc, exv);
  return shape;
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

  // Contiguous variables that are represented with a box
  // may require the shape to be extracted from the box (i.e. evx),
  // because they itself may not have shape specified.
  // This happens during late propagationg of contiguous
  // attribute, e.g.:
  // %9:2 = hlfir.declare %6
  //     {fortran_attrs = #fir.var_attrs<contiguous>} :
  //     (!fir.box<!fir.array<?x?x...>>) ->
  //     (!fir.box<!fir.array<?x?x...>>, !fir.box<!fir.array<?x?x...>>)
  // The extended value is an ArrayBoxValue with base being
  // the raw address of the array.
  if (auto variableInterface = entity.getIfVariableInterface()) {
    mlir::Value shape = variableInterface.getShape();
    if (mlir::isa<fir::BaseBoxType>(fir::getBase(exv).getType()) ||
        !mlir::isa<fir::BaseBoxType>(entity.getType()) ||
        // Still use the variable's shape if it is present.
        // If it only specifies a shift, then we have to create
        // a shape from the exv.
        (shape && (shape.getDefiningOp<fir::ShapeShiftOp>() ||
                   shape.getDefiningOp<fir::ShapeOp>())))
      return {fir::getBase(exv),
              asEmboxShape(loc, builder, exv, variableInterface.getShape())};
  }
  return {fir::getBase(exv), builder.createShape(loc, exv)};
}

hlfir::Entity hlfir::derefPointersAndAllocatables(mlir::Location loc,
                                                  fir::FirOpBuilder &builder,
                                                  Entity entity) {
  if (entity.isMutableBox()) {
    hlfir::Entity boxLoad{fir::LoadOp::create(builder, loc, entity)};
    if (entity.isScalar()) {
      if (!entity.isPolymorphic() && !entity.hasLengthParameters())
        return hlfir::Entity{fir::BoxAddrOp::create(builder, loc, boxLoad)};
      mlir::Type elementType = boxLoad.getFortranElementType();
      if (auto charType = mlir::dyn_cast<fir::CharacterType>(elementType)) {
        mlir::Value base = fir::BoxAddrOp::create(builder, loc, boxLoad);
        if (charType.hasConstantLen())
          return hlfir::Entity{base};
        mlir::Value len = genCharacterVariableLength(loc, builder, entity);
        auto boxCharType =
            fir::BoxCharType::get(builder.getContext(), charType.getFKind());
        return hlfir::Entity{
            fir::EmboxCharOp::create(builder, loc, boxCharType, base, len)
                .getResult()};
      }
    }
    // Otherwise, the entity is either an array, a polymorphic entity, or a
    // derived type with length parameters. All these entities require a fir.box
    // or fir.class to hold bounds, dynamic type or length parameter
    // information. Keep them boxed.
    return boxLoad;
  } else if (entity.isProcedurePointer()) {
    return hlfir::Entity{fir::LoadOp::create(builder, loc, entity)};
  }
  return entity;
}

mlir::Type hlfir::getVariableElementType(hlfir::Entity variable) {
  assert(variable.isVariable() && "entity must be a variable");
  if (variable.isScalar())
    return variable.getType();
  mlir::Type eleTy = variable.getFortranElementType();
  const bool isVolatile = fir::isa_volatile_type(variable.getType());
  if (variable.isPolymorphic())
    return fir::ClassType::get(eleTy, isVolatile);
  if (auto charType = mlir::dyn_cast<fir::CharacterType>(eleTy)) {
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());
  } else if (fir::isRecordWithTypeParameters(eleTy)) {
    return fir::BoxType::get(eleTy, isVolatile);
  }
  return fir::ReferenceType::get(eleTy, isVolatile);
}

mlir::Type hlfir::getEntityElementType(hlfir::Entity entity) {
  if (entity.isVariable())
    return getVariableElementType(entity);
  if (entity.isScalar())
    return entity.getType();
  auto exprType = mlir::dyn_cast<hlfir::ExprType>(entity.getType());
  assert(exprType && "array value must be an hlfir.expr");
  return exprType.getElementExprType();
}

static hlfir::ExprType getArrayExprType(mlir::Type elementType,
                                        mlir::Value shape, bool isPolymorphic) {
  unsigned rank = mlir::cast<fir::ShapeType>(shape.getType()).getRank();
  hlfir::ExprType::Shape typeShape(rank, hlfir::ExprType::getUnknownExtent());
  if (auto shapeOp = shape.getDefiningOp<fir::ShapeOp>())
    for (auto extent : llvm::enumerate(shapeOp.getExtents()))
      if (auto cstExtent = fir::getIntIfConstant(extent.value()))
        typeShape[extent.index()] = *cstExtent;
  return hlfir::ExprType::get(elementType.getContext(), typeShape, elementType,
                              isPolymorphic);
}

hlfir::ElementalOp hlfir::genElementalOp(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::Type elementType,
    mlir::Value shape, mlir::ValueRange typeParams,
    const ElementalKernelGenerator &genKernel, bool isUnordered,
    mlir::Value polymorphicMold, mlir::Type exprType) {
  if (!exprType)
    exprType = getArrayExprType(elementType, shape, !!polymorphicMold);
  auto elementalOp = hlfir::ElementalOp::create(
      builder, loc, exprType, shape, polymorphicMold, typeParams, isUnordered);
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(elementalOp.getBody());
  mlir::Value elementResult = genKernel(loc, builder, elementalOp.getIndices());
  // Numerical and logical scalars may be lowered to another type than the
  // Fortran expression type (e.g i1 instead of fir.logical). Array expression
  // values are typed according to their Fortran type. Insert a cast if needed
  // here.
  if (fir::isa_trivial(elementResult.getType()))
    elementResult = builder.createConvert(loc, elementType, elementResult);
  hlfir::YieldElementOp::create(builder, loc, elementResult);
  builder.restoreInsertionPoint(insertPt);
  return elementalOp;
}

// TODO: we do not actually need to clone the YieldElementOp,
// because returning its getElementValue() operand should be enough
// for all callers of this function.
hlfir::YieldElementOp
hlfir::inlineElementalOp(mlir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::ElementalOp elemental,
                         mlir::ValueRange oneBasedIndices) {
  // hlfir.elemental region is a SizedRegion<1>.
  assert(elemental.getRegion().hasOneBlock() &&
         "expect elemental region to have one block");
  mlir::IRMapping mapper;
  mapper.map(elemental.getIndices(), oneBasedIndices);
  mlir::Operation *newOp;
  for (auto &op : elemental.getRegion().back().getOperations())
    newOp = builder.clone(op, mapper);
  auto yield = mlir::dyn_cast_or_null<hlfir::YieldElementOp>(newOp);
  assert(yield && "last ElementalOp operation must be am hlfir.yield_element");
  return yield;
}

mlir::Value hlfir::inlineElementalOp(
    mlir::Location loc, fir::FirOpBuilder &builder,
    hlfir::ElementalOpInterface elemental, mlir::ValueRange oneBasedIndices,
    mlir::IRMapping &mapper,
    const std::function<bool(hlfir::ElementalOp)> &mustRecursivelyInline) {
  mlir::Region &region = elemental.getElementalRegion();
  // hlfir.elemental region is a SizedRegion<1>.
  assert(region.hasOneBlock() && "elemental region must have one block");
  mapper.map(elemental.getIndices(), oneBasedIndices);
  for (auto &op : region.front().without_terminator()) {
    if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op))
      if (auto appliedElemental =
              apply.getExpr().getDefiningOp<hlfir::ElementalOp>())
        if (mustRecursivelyInline(appliedElemental)) {
          llvm::SmallVector<mlir::Value> clonedApplyIndices;
          for (auto indice : apply.getIndices())
            clonedApplyIndices.push_back(mapper.lookupOrDefault(indice));
          hlfir::ElementalOpInterface elementalIface =
              mlir::cast<hlfir::ElementalOpInterface>(
                  appliedElemental.getOperation());
          mlir::Value inlined = inlineElementalOp(loc, builder, elementalIface,
                                                  clonedApplyIndices, mapper,
                                                  mustRecursivelyInline);
          mapper.map(apply.getResult(), inlined);
          continue;
        }
    (void)builder.clone(op, mapper);
  }
  return mapper.lookupOrDefault(elemental.getElementEntity());
}

hlfir::LoopNest hlfir::genLoopNest(mlir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   mlir::ValueRange extents, bool isUnordered,
                                   bool emitWorkshareLoop,
                                   bool couldVectorize) {
  emitWorkshareLoop = emitWorkshareLoop && isUnordered;
  hlfir::LoopNest loopNest;
  assert(!extents.empty() && "must have at least one extent");
  mlir::OpBuilder::InsertionGuard guard(builder);
  loopNest.oneBasedIndices.assign(extents.size(), mlir::Value{});
  // Build loop nest from column to row.
  auto one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Type indexType = builder.getIndexType();
  if (emitWorkshareLoop) {
    auto wslw = mlir::omp::WorkshareLoopWrapperOp::create(builder, loc);
    loopNest.outerOp = wslw;
    builder.createBlock(&wslw.getRegion());
    mlir::omp::LoopNestOperands lnops;
    lnops.loopInclusive = builder.getUnitAttr();
    for (auto extent : llvm::reverse(extents)) {
      lnops.loopLowerBounds.push_back(one);
      lnops.loopUpperBounds.push_back(extent);
      lnops.loopSteps.push_back(one);
    }
    auto lnOp = mlir::omp::LoopNestOp::create(builder, loc, lnops);
    mlir::Block *block = builder.createBlock(&lnOp.getRegion());
    for (auto extent : llvm::reverse(extents))
      block->addArgument(extent.getType(), extent.getLoc());
    loopNest.body = block;
    mlir::omp::YieldOp::create(builder, loc);
    for (unsigned dim = 0; dim < extents.size(); dim++)
      loopNest.oneBasedIndices[extents.size() - dim - 1] =
          lnOp.getRegion().front().getArgument(dim);
  } else {
    unsigned dim = extents.size() - 1;
    for (auto extent : llvm::reverse(extents)) {
      auto ub = builder.createConvert(loc, indexType, extent);
      auto doLoop =
          fir::DoLoopOp::create(builder, loc, one, ub, one, isUnordered);
      if (!couldVectorize) {
        mlir::LLVM::LoopVectorizeAttr va{mlir::LLVM::LoopVectorizeAttr::get(
            builder.getContext(),
            /*disable=*/builder.getBoolAttr(true), {}, {}, {}, {}, {}, {})};
        mlir::LLVM::LoopAnnotationAttr la = mlir::LLVM::LoopAnnotationAttr::get(
            builder.getContext(), {}, /*vectorize=*/va, {}, /*unroll*/ {},
            /*unroll_and_jam*/ {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});
        doLoop.setLoopAnnotationAttr(la);
      }
      loopNest.body = doLoop.getBody();
      builder.setInsertionPointToStart(loopNest.body);
      // Reverse the indices so they are in column-major order.
      loopNest.oneBasedIndices[dim--] = doLoop.getInductionVar();
      if (!loopNest.outerOp)
        loopNest.outerOp = doLoop;
    }
  }
  return loopNest;
}

llvm::SmallVector<mlir::Value> hlfir::genLoopNestWithReductions(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::ValueRange extents,
    mlir::ValueRange reductionInits, const ReductionLoopBodyGenerator &genBody,
    bool isUnordered) {
  assert(!extents.empty() && "must have at least one extent");
  // Build loop nest from column to row.
  auto one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Type indexType = builder.getIndexType();
  unsigned dim = extents.size() - 1;
  fir::DoLoopOp outerLoop = nullptr;
  fir::DoLoopOp parentLoop = nullptr;
  llvm::SmallVector<mlir::Value> oneBasedIndices;
  oneBasedIndices.resize(dim + 1);
  for (auto extent : llvm::reverse(extents)) {
    auto ub = builder.createConvert(loc, indexType, extent);

    // The outermost loop takes reductionInits as the initial
    // values of its iter-args.
    // A child loop takes its iter-args from the region iter-args
    // of its parent loop.
    fir::DoLoopOp doLoop;
    if (!parentLoop) {
      doLoop = fir::DoLoopOp::create(builder, loc, one, ub, one, isUnordered,
                                     /*finalCountValue=*/false, reductionInits);
    } else {
      doLoop = fir::DoLoopOp::create(builder, loc, one, ub, one, isUnordered,
                                     /*finalCountValue=*/false,
                                     parentLoop.getRegionIterArgs());
      if (!reductionInits.empty()) {
        // Return the results of the child loop from its parent loop.
        fir::ResultOp::create(builder, loc, doLoop.getResults());
      }
    }

    builder.setInsertionPointToStart(doLoop.getBody());
    // Reverse the indices so they are in column-major order.
    oneBasedIndices[dim--] = doLoop.getInductionVar();
    if (!outerLoop)
      outerLoop = doLoop;
    parentLoop = doLoop;
  }

  llvm::SmallVector<mlir::Value> reductionValues;
  reductionValues =
      genBody(loc, builder, oneBasedIndices, parentLoop.getRegionIterArgs());
  builder.setInsertionPointToEnd(parentLoop.getBody());
  if (!reductionValues.empty())
    fir::ResultOp::create(builder, loc, reductionValues);
  builder.setInsertionPointAfter(outerLoop);
  return outerLoop->getResults();
}

template <typename Lambda>
static fir::ExtendedValue
conditionallyEvaluate(mlir::Location loc, fir::FirOpBuilder &builder,
                      mlir::Value condition, const Lambda &genIfTrue) {
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();

  // Evaluate in some region that will be moved into the actual ifOp (the actual
  // ifOp can only be created when the result types are known).
  auto badIfOp = fir::IfOp::create(builder, loc, condition.getType(), condition,
                                   /*withElseRegion=*/false);
  mlir::Block *preparationBlock = &badIfOp.getThenRegion().front();
  builder.setInsertionPointToStart(preparationBlock);
  fir::ExtendedValue result = genIfTrue();
  fir::ResultOp resultOp = result.match(
      [&](const fir::CharBoxValue &box) -> fir::ResultOp {
        return fir::ResultOp::create(
            builder, loc, mlir::ValueRange{box.getAddr(), box.getLen()});
      },
      [&](const mlir::Value &addr) -> fir::ResultOp {
        return fir::ResultOp::create(builder, loc, addr);
      },
      [&](const auto &) -> fir::ResultOp {
        TODO(loc, "unboxing non scalar optional fir.box");
      });
  builder.restoreInsertionPoint(insertPt);

  // Create actual fir.if operation.
  auto ifOp =
      fir::IfOp::create(builder, loc, resultOp->getOperandTypes(), condition,
                        /*withElseRegion=*/true);
  // Move evaluation into Then block,
  preparationBlock->moveBefore(&ifOp.getThenRegion().back());
  ifOp.getThenRegion().back().erase();
  // Create absent result in the Else block.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  llvm::SmallVector<mlir::Value> absentValues;
  for (mlir::Type resTy : ifOp->getResultTypes()) {
    if (fir::isa_ref_type(resTy) || fir::isa_box_type(resTy))
      absentValues.emplace_back(fir::AbsentOp::create(builder, loc, resTy));
    else
      absentValues.emplace_back(fir::ZeroOp::create(builder, loc, resTy));
  }
  fir::ResultOp::create(builder, loc, absentValues);
  badIfOp->erase();

  // Build fir::ExtendedValue from the result values.
  builder.setInsertionPointAfter(ifOp);
  return result.match(
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
        return fir::CharBoxValue{ifOp.getResult(0), ifOp.getResult(1)};
      },
      [&](const mlir::Value &) -> fir::ExtendedValue {
        return ifOp.getResult(0);
      },
      [&](const auto &) -> fir::ExtendedValue {
        TODO(loc, "unboxing non scalar optional fir.box");
      });
}

static fir::ExtendedValue translateVariableToExtendedValue(
    mlir::Location loc, fir::FirOpBuilder &builder, hlfir::Entity variable,
    bool forceHlfirBase = false, bool contiguousHint = false,
    bool keepScalarOptionalBoxed = false) {
  assert(variable.isVariable() && "must be a variable");
  // When going towards FIR, use the original base value to avoid
  // introducing descriptors at runtime when they are not required.
  // This is not done for assumed-rank since the fir::ExtendedValue cannot
  // held the related lower bounds in an vector. The lower bounds of the
  // descriptor must always be used instead.

  mlir::Value base = (forceHlfirBase || variable.isAssumedRank())
                         ? variable.getBase()
                         : variable.getFirBase();
  if (variable.isMutableBox())
    return fir::MutableBoxValue(base, getExplicitTypeParams(variable),
                                fir::MutableProperties{});

  if (mlir::isa<fir::BaseBoxType>(base.getType())) {
    const bool contiguous = variable.isSimplyContiguous() || contiguousHint;
    const bool isAssumedRank = variable.isAssumedRank();
    if (!contiguous || variable.isPolymorphic() ||
        variable.isDerivedWithLengthParameters() || isAssumedRank) {
      llvm::SmallVector<mlir::Value> nonDefaultLbounds;
      if (!isAssumedRank)
        nonDefaultLbounds = getNonDefaultLowerBounds(loc, builder, variable);
      return fir::BoxValue(base, nonDefaultLbounds,
                           getExplicitTypeParams(variable));
    }
    if (variable.mayBeOptional()) {
      if (!keepScalarOptionalBoxed && variable.isScalar()) {
        mlir::Value isPresent = fir::IsPresentOp::create(
            builder, loc, builder.getI1Type(), variable);
        return conditionallyEvaluate(
            loc, builder, isPresent, [&]() -> fir::ExtendedValue {
              mlir::Value base = genVariableRawAddress(loc, builder, variable);
              if (variable.isCharacter()) {
                mlir::Value len =
                    genCharacterVariableLength(loc, builder, variable);
                return fir::CharBoxValue{base, len};
              }
              return base;
            });
      }
      llvm::SmallVector<mlir::Value> nonDefaultLbounds =
          getNonDefaultLowerBounds(loc, builder, variable);
      return fir::BoxValue(base, nonDefaultLbounds,
                           getExplicitTypeParams(variable));
    }
    // Otherwise, the variable can be represented in a fir::ExtendedValue
    // without the overhead of a fir.box.
    base = genVariableRawAddress(loc, builder, variable);
  }

  if (variable.isScalar()) {
    if (variable.isCharacter()) {
      if (mlir::isa<fir::BoxCharType>(base.getType()))
        return genUnboxChar(loc, builder, base);
      mlir::Value len = genCharacterVariableLength(loc, builder, variable);
      return fir::CharBoxValue{base, len};
    }
    return base;
  }
  llvm::SmallVector<mlir::Value> extents;
  llvm::SmallVector<mlir::Value> nonDefaultLbounds;
  if (mlir::isa<fir::BaseBoxType>(variable.getType()) &&
      !variable.getIfVariableInterface() &&
      variable.mayHaveNonDefaultLowerBounds()) {
    // This special case avoids generating two sets of identical
    // fir.box_dim to get both the lower bounds and extents.
    fir::factory::genDimInfoFromBox(builder, loc, variable, &nonDefaultLbounds,
                                    &extents, /*strides=*/nullptr);
  } else {
    extents = getVariableExtents(loc, builder, variable);
    nonDefaultLbounds = getNonDefaultLowerBounds(loc, builder, variable);
  }
  if (variable.isCharacter())
    return fir::CharArrayBoxValue{
        base, genCharacterVariableLength(loc, builder, variable), extents,
        nonDefaultLbounds};
  return fir::ArrayBoxValue{base, extents, nonDefaultLbounds};
}

fir::ExtendedValue
hlfir::translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                                fir::FortranVariableOpInterface var,
                                bool forceHlfirBase) {
  return translateVariableToExtendedValue(loc, builder, var, forceHlfirBase);
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::translateToExtendedValue(mlir::Location loc, fir::FirOpBuilder &builder,
                                hlfir::Entity entity, bool contiguousHint,
                                bool keepScalarOptionalBoxed) {
  if (entity.isVariable())
    return {translateVariableToExtendedValue(loc, builder, entity, false,
                                             contiguousHint,
                                             keepScalarOptionalBoxed),
            std::nullopt};

  if (entity.isProcedure()) {
    if (fir::isCharacterProcedureTuple(entity.getType())) {
      auto [boxProc, len] = fir::factory::extractCharacterProcedureTuple(
          builder, loc, entity, /*openBoxProc=*/false);
      return {fir::CharBoxValue{boxProc, len}, std::nullopt};
    }
    return {static_cast<mlir::Value>(entity), std::nullopt};
  }

  if (mlir::isa<hlfir::ExprType>(entity.getType())) {
    mlir::NamedAttribute byRefAttr = fir::getAdaptToByRefAttr(builder);
    hlfir::AssociateOp associate = hlfir::genAssociateExpr(
        loc, builder, entity, entity.getType(), "", byRefAttr);
    auto *bldr = &builder;
    hlfir::CleanupFunction cleanup = [bldr, loc, associate]() -> void {
      hlfir::EndAssociateOp::create(*bldr, loc, associate);
    };
    hlfir::Entity temp{associate.getBase()};
    return {translateToExtendedValue(loc, builder, temp).first, cleanup};
  }
  return {{static_cast<mlir::Value>(entity)}, {}};
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::convertToValue(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity entity) {
  // Load scalar references to integer, logical, real, or complex value
  // to an mlir value, dereference allocatable and pointers, and get rid
  // of fir.box that are not needed or create a copy into contiguous memory.
  auto derefedAndLoadedEntity = loadTrivialScalar(loc, builder, entity);
  return translateToExtendedValue(loc, builder, derefedAndLoadedEntity);
}

static fir::ExtendedValue placeTrivialInMemory(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               mlir::Value val,
                                               mlir::Type targetType) {
  auto temp = builder.createTemporary(loc, targetType);
  if (targetType != val.getType())
    builder.createStoreWithConvert(loc, val, temp);
  else
    fir::StoreOp::create(builder, loc, val, temp);
  return temp;
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::convertToBox(mlir::Location loc, fir::FirOpBuilder &builder,
                    hlfir::Entity entity, mlir::Type targetType) {
  // fir::factory::createBoxValue is not meant to deal with procedures.
  // Dereference procedure pointers here.
  if (entity.isProcedurePointer())
    entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);

  auto [exv, cleanup] =
      translateToExtendedValue(loc, builder, entity, /*contiguousHint=*/false,
                               /*keepScalarOptionalBoxed=*/true);
  // Procedure entities should not go through createBoxValue that embox
  // object entities. Return the fir.boxproc directly.
  if (entity.isProcedure())
    return {exv, cleanup};
  mlir::Value base = fir::getBase(exv);
  if (fir::isa_trivial(base.getType()))
    exv = placeTrivialInMemory(loc, builder, base, targetType);
  fir::BoxValue box = fir::factory::createBoxValue(builder, loc, exv);
  return {box, cleanup};
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::convertToAddress(mlir::Location loc, fir::FirOpBuilder &builder,
                        hlfir::Entity entity, mlir::Type targetType) {
  hlfir::Entity derefedEntity =
      hlfir::derefPointersAndAllocatables(loc, builder, entity);
  auto [exv, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, derefedEntity);
  mlir::Value base = fir::getBase(exv);
  if (fir::isa_trivial(base.getType()))
    exv = placeTrivialInMemory(loc, builder, base, targetType);
  return {exv, cleanup};
}

/// Clone:
/// ```
/// hlfir.elemental_addr %shape : !fir.shape<1> {
///   ^bb0(%i : index)
///    .....
///    %hlfir.yield %scalarAddress : fir.ref<T>
/// }
/// ```
//
/// into
///
/// ```
/// %expr = hlfir.elemental %shape : (!fir.shape<1>) -> hlfir.expr<?xT> {
///   ^bb0(%i : index)
///    .....
///    %value = fir.load %scalarAddress : fir.ref<T>
///    %hlfir.yield_element %value : T
///  }
/// ```
hlfir::ElementalOp
hlfir::cloneToElementalOp(mlir::Location loc, fir::FirOpBuilder &builder,
                          hlfir::ElementalAddrOp elementalAddrOp) {
  hlfir::Entity scalarAddress =
      hlfir::Entity{mlir::cast<hlfir::YieldOp>(
                        elementalAddrOp.getBody().back().getTerminator())
                        .getEntity()};
  llvm::SmallVector<mlir::Value, 1> typeParams;
  hlfir::genLengthParameters(loc, builder, scalarAddress, typeParams);

  builder.setInsertionPointAfter(elementalAddrOp);
  auto genKernel = [&](mlir::Location l, fir::FirOpBuilder &b,
                       mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
    mlir::IRMapping mapper;
    mapper.map(elementalAddrOp.getIndices(), oneBasedIndices);
    mlir::Operation *newOp = nullptr;
    for (auto &op : elementalAddrOp.getBody().back().getOperations())
      newOp = b.clone(op, mapper);
    auto newYielOp = mlir::dyn_cast_or_null<hlfir::YieldOp>(newOp);
    assert(newYielOp && "hlfir.elemental_addr is ill formed");
    hlfir::Entity newAddr{newYielOp.getEntity()};
    newYielOp->erase();
    return hlfir::loadTrivialScalar(l, b, newAddr);
  };
  mlir::Type elementType = scalarAddress.getFortranElementType();
  return hlfir::genElementalOp(
      loc, builder, elementType, elementalAddrOp.getShape(), typeParams,
      genKernel, !elementalAddrOp.isOrdered(), elementalAddrOp.getMold());
}

bool hlfir::elementalOpMustProduceTemp(hlfir::ElementalOp elemental) {
  for (mlir::Operation *useOp : elemental->getUsers())
    if (auto destroy = mlir::dyn_cast<hlfir::DestroyOp>(useOp))
      if (destroy.mustFinalizeExpr())
        return true;

  return false;
}

std::pair<hlfir::Entity, mlir::Value>
hlfir::createTempFromMold(mlir::Location loc, fir::FirOpBuilder &builder,
                          hlfir::Entity mold) {
  assert(!mold.isAssumedRank() &&
         "cannot create temporary from assumed-rank mold");
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};

  mlir::Value shape{};
  llvm::SmallVector<mlir::Value> extents;
  if (mold.isArray()) {
    shape = hlfir::genShape(loc, builder, mold);
    extents = hlfir::getExplicitExtentsFromShape(shape, builder);
  }

  bool useStack = !mold.isArray() && !mold.isPolymorphic();
  auto genTempDeclareOp =
      [](fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value memref,
         llvm::StringRef name, mlir::Value shape,
         llvm::ArrayRef<mlir::Value> typeParams,
         fir::FortranVariableFlagsAttr attrs) -> mlir::Value {
    auto declareOp =
        hlfir::DeclareOp::create(builder, loc, memref, name, shape, typeParams,
                                 /*dummy_scope=*/nullptr, attrs);
    return declareOp.getBase();
  };

  auto [base, isHeapAlloc] = builder.createAndDeclareTemp(
      loc, mold.getElementOrSequenceType(), shape, extents, lenParams,
      genTempDeclareOp, mold.isPolymorphic() ? mold.getBase() : nullptr,
      useStack, tmpName);
  return {hlfir::Entity{base}, builder.createBool(loc, isHeapAlloc)};
}

hlfir::Entity hlfir::createStackTempFromMold(mlir::Location loc,
                                             fir::FirOpBuilder &builder,
                                             hlfir::Entity mold) {
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};
  mlir::Value alloc;
  mlir::Value shape{};
  fir::FortranVariableFlagsAttr declAttrs;

  if (mold.isPolymorphic()) {
    // genAllocatableApplyMold does heap allocation
    TODO(loc, "createStackTempFromMold for polymorphic type");
  } else if (mold.isArray()) {
    mlir::Type sequenceType =
        hlfir::getFortranElementOrSequenceType(mold.getType());
    shape = hlfir::genShape(loc, builder, mold);
    auto extents = hlfir::getIndexExtents(loc, builder, shape);
    alloc =
        builder.createTemporary(loc, sequenceType, tmpName, extents, lenParams);
  } else {
    alloc = builder.createTemporary(loc, mold.getFortranElementType(), tmpName,
                                    /*shape=*/{}, lenParams);
  }
  auto declareOp =
      hlfir::DeclareOp::create(builder, loc, alloc, tmpName, shape, lenParams,
                               /*dummy_scope=*/nullptr, declAttrs);
  return hlfir::Entity{declareOp.getBase()};
}

hlfir::EntityWithAttributes
hlfir::convertCharacterKind(mlir::Location loc, fir::FirOpBuilder &builder,
                            hlfir::Entity scalarChar, int toKind) {
  auto src = hlfir::convertToAddress(loc, builder, scalarChar,
                                     scalarChar.getFortranElementType());
  assert(src.first.getCharBox() && "must be scalar character");
  fir::CharBoxValue res = fir::factory::convertCharacterKind(
      builder, loc, *src.first.getCharBox(), toKind);
  if (src.second.has_value())
    src.second.value()();

  return hlfir::EntityWithAttributes{hlfir::DeclareOp::create(
      builder, loc, res.getAddr(), ".temp.kindconvert", /*shape=*/nullptr,
      /*typeparams=*/mlir::ValueRange{res.getLen()},
      /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{})};
}

std::pair<hlfir::Entity, std::optional<hlfir::CleanupFunction>>
hlfir::genTypeAndKindConvert(mlir::Location loc, fir::FirOpBuilder &builder,
                             hlfir::Entity source, mlir::Type toType,
                             bool preserveLowerBounds) {
  mlir::Type fromType = source.getFortranElementType();
  toType = hlfir::getFortranElementType(toType);
  if (!toType || fromType == toType ||
      !(fir::isa_trivial(toType) || mlir::isa<fir::CharacterType>(toType)))
    return {source, std::nullopt};

  std::optional<int> toKindCharConvert;
  if (auto toCharTy = mlir::dyn_cast<fir::CharacterType>(toType)) {
    if (auto fromCharTy = mlir::dyn_cast<fir::CharacterType>(fromType))
      if (toCharTy.getFKind() != fromCharTy.getFKind()) {
        toKindCharConvert = toCharTy.getFKind();
        // Preserve source length (padding/truncation will occur in assignment
        // if needed).
        toType = fir::CharacterType::get(
            fromType.getContext(), toCharTy.getFKind(), fromCharTy.getLen());
      }
    // Do not convert in case of character length mismatch only, hlfir.assign
    // deals with it.
    if (!toKindCharConvert)
      return {source, std::nullopt};
  }

  if (source.getRank() == 0) {
    mlir::Value cast = toKindCharConvert
                           ? mlir::Value{hlfir::convertCharacterKind(
                                 loc, builder, source, *toKindCharConvert)}
                           : builder.convertWithSemantics(loc, toType, source);
    return {hlfir::Entity{cast}, std::nullopt};
  }

  mlir::Value shape = hlfir::genShape(loc, builder, source);
  auto genKernel = [source, toType, toKindCharConvert](
                       mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
    auto elementPtr =
        hlfir::getElementAt(loc, builder, source, oneBasedIndices);
    auto val = hlfir::loadTrivialScalar(loc, builder, elementPtr);
    if (toKindCharConvert)
      return hlfir::convertCharacterKind(loc, builder, val, *toKindCharConvert);
    return hlfir::EntityWithAttributes{
        builder.convertWithSemantics(loc, toType, val)};
  };
  llvm::SmallVector<mlir::Value, 1> lenParams;
  hlfir::genLengthParameters(loc, builder, source, lenParams);
  mlir::Value convertedRhs =
      hlfir::genElementalOp(loc, builder, toType, shape, lenParams, genKernel,
                            /*isUnordered=*/true);

  if (preserveLowerBounds && source.mayHaveNonDefaultLowerBounds()) {
    hlfir::AssociateOp associate =
        genAssociateExpr(loc, builder, hlfir::Entity{convertedRhs},
                         convertedRhs.getType(), ".tmp.keeplbounds");
    fir::ShapeOp shapeOp = associate.getShape().getDefiningOp<fir::ShapeOp>();
    assert(shapeOp && "associate shape must be a fir.shape");
    const unsigned rank = shapeOp.getExtents().size();
    llvm::SmallVector<mlir::Value> lbAndExtents;
    for (unsigned dim = 0; dim < rank; ++dim) {
      lbAndExtents.push_back(hlfir::genLBound(loc, builder, source, dim));
      lbAndExtents.push_back(shapeOp.getExtents()[dim]);
    }
    auto shapeShiftType = fir::ShapeShiftType::get(builder.getContext(), rank);
    mlir::Value shapeShift =
        fir::ShapeShiftOp::create(builder, loc, shapeShiftType, lbAndExtents);
    auto declareOp = hlfir::DeclareOp::create(
        builder, loc, associate.getFirBase(), *associate.getUniqName(),
        shapeShift, associate.getTypeparams(), /*dummy_scope=*/nullptr,
        /*flags=*/fir::FortranVariableFlagsAttr{});
    hlfir::Entity castWithLbounds =
        mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation());
    fir::FirOpBuilder *bldr = &builder;
    auto cleanup = [loc, bldr, convertedRhs, associate]() {
      hlfir::EndAssociateOp::create(*bldr, loc, associate);
      hlfir::DestroyOp::create(*bldr, loc, convertedRhs);
    };
    return {castWithLbounds, cleanup};
  }

  fir::FirOpBuilder *bldr = &builder;
  auto cleanup = [loc, bldr, convertedRhs]() {
    hlfir::DestroyOp::create(*bldr, loc, convertedRhs);
  };
  return {hlfir::Entity{convertedRhs}, cleanup};
}

std::pair<hlfir::Entity, bool> hlfir::computeEvaluateOpInNewTemp(
    mlir::Location loc, fir::FirOpBuilder &builder,
    hlfir::EvaluateInMemoryOp evalInMem, mlir::Value shape,
    mlir::ValueRange typeParams) {
  llvm::StringRef tmpName{".tmp.expr_result"};
  llvm::SmallVector<mlir::Value> extents =
      hlfir::getIndexExtents(loc, builder, shape);
  mlir::Type baseType =
      hlfir::getFortranElementOrSequenceType(evalInMem.getType());
  bool heapAllocated = fir::hasDynamicSize(baseType);
  // Note: temporaries are stack allocated here when possible (do not require
  // stack save/restore) because flang has always stack allocated function
  // results.
  mlir::Value temp = heapAllocated
                         ? builder.createHeapTemporary(loc, baseType, tmpName,
                                                       extents, typeParams)
                         : builder.createTemporary(loc, baseType, tmpName,
                                                   extents, typeParams);
  mlir::Value innerMemory = evalInMem.getMemory();
  temp = builder.createConvert(loc, innerMemory.getType(), temp);
  auto declareOp = hlfir::DeclareOp::create(
      builder, loc, temp, tmpName, shape, typeParams,
      /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});
  computeEvaluateOpIn(loc, builder, evalInMem, declareOp.getOriginalBase());
  return {hlfir::Entity{declareOp.getBase()}, /*heapAllocated=*/heapAllocated};
}

void hlfir::computeEvaluateOpIn(mlir::Location loc, fir::FirOpBuilder &builder,
                                hlfir::EvaluateInMemoryOp evalInMem,
                                mlir::Value storage) {
  mlir::Value innerMemory = evalInMem.getMemory();
  mlir::Value storageCast =
      builder.createConvert(loc, innerMemory.getType(), storage);
  mlir::IRMapping mapper;
  mapper.map(innerMemory, storageCast);
  for (auto &op : evalInMem.getBody().front().without_terminator())
    builder.clone(op, mapper);
  return;
}

hlfir::Entity hlfir::loadElementAt(mlir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   hlfir::Entity entity,
                                   mlir::ValueRange oneBasedIndices) {
  return loadTrivialScalar(loc, builder,
                           getElementAt(loc, builder, entity, oneBasedIndices));
}

llvm::SmallVector<mlir::Value, Fortran::common::maxRank>
hlfir::genExtentsVector(mlir::Location loc, fir::FirOpBuilder &builder,
                        hlfir::Entity entity) {
  entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
  mlir::Value shape = hlfir::genShape(loc, builder, entity);
  llvm::SmallVector<mlir::Value, Fortran::common::maxRank> extents =
      hlfir::getExplicitExtentsFromShape(shape, builder);
  if (shape.getUses().empty())
    shape.getDefiningOp()->erase();
  return extents;
}

hlfir::Entity hlfir::gen1DSection(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  hlfir::Entity array, int64_t dim,
                                  mlir::ArrayRef<mlir::Value> lbounds,
                                  mlir::ArrayRef<mlir::Value> extents,
                                  mlir::ValueRange oneBasedIndices,
                                  mlir::ArrayRef<mlir::Value> typeParams) {
  assert(array.isVariable() && "array must be a variable");
  assert(dim > 0 && dim <= array.getRank() && "invalid dim number");
  mlir::Value one =
      builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  hlfir::DesignateOp::Subscripts subscripts;
  unsigned indexId = 0;
  for (int i = 0; i < array.getRank(); ++i) {
    if (i == dim - 1) {
      mlir::Value ubound = genUBound(loc, builder, lbounds[i], extents[i], one);
      subscripts.emplace_back(
          hlfir::DesignateOp::Triplet{lbounds[i], ubound, one});
    } else {
      mlir::Value index =
          genUBound(loc, builder, lbounds[i], oneBasedIndices[indexId++], one);
      subscripts.emplace_back(index);
    }
  }
  mlir::Value sectionShape =
      fir::ShapeOp::create(builder, loc, extents[dim - 1]);

  // The result type is one of:
  //   !fir.box/class<!fir.array<NxT>>
  //   !fir.box/class<!fir.array<?xT>>
  //
  // We could use !fir.ref<!fir.array<NxT>> when the whole dimension's
  // size is known and it is the leading dimension, but let it be simple
  // for the time being.
  auto seqType =
      mlir::cast<fir::SequenceType>(array.getElementOrSequenceType());
  int64_t dimExtent = seqType.getShape()[dim - 1];
  mlir::Type sectionType =
      fir::SequenceType::get({dimExtent}, seqType.getEleTy());
  sectionType = fir::wrapInClassOrBoxType(sectionType, array.isPolymorphic());

  auto designate = hlfir::DesignateOp::create(
      builder, loc, sectionType, array, /*component=*/"",
      /*componentShape=*/nullptr, subscripts,
      /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
      sectionShape, typeParams);
  return hlfir::Entity{designate.getResult()};
}

bool hlfir::designatePreservesContinuity(hlfir::DesignateOp op) {
  if (op.getComponent() || op.getComplexPart() || !op.getSubstring().empty())
    return false;
  auto subscripts = op.getIndices();
  unsigned i = 0;
  for (auto isTriplet : llvm::enumerate(op.getIsTriplet())) {
    // TODO: we should allow any number of leading triplets
    // that describe a whole dimension slice, then one optional
    // triplet describing potentially partial dimension slice,
    // then any number of non-triplet subscripts.
    // For the time being just allow a single leading
    // triplet and then any number of non-triplet subscripts.
    if (isTriplet.value()) {
      if (isTriplet.index() != 0) {
        return false;
      } else {
        i += 2;
        mlir::Value step = subscripts[i++];
        auto constantStep = fir::getIntIfConstant(step);
        if (!constantStep || *constantStep != 1)
          return false;
      }
    } else {
      ++i;
    }
  }
  return true;
}

bool hlfir::isSimplyContiguous(mlir::Value base, bool checkWhole) {
  hlfir::Entity entity{base};
  if (entity.isSimplyContiguous())
    return true;

  // Look at the definition.
  mlir::Operation *def = base.getDefiningOp();
  if (!def)
    return false;

  return mlir::TypeSwitch<mlir::Operation *, bool>(def)
      .Case<fir::EmboxOp>(
          [&](auto op) { return fir::isContiguousEmbox(op, checkWhole); })
      .Case<fir::ReboxOp>([&](auto op) {
        hlfir::Entity box{op.getBox()};
        return fir::reboxPreservesContinuity(
                   op, box.mayHaveNonDefaultLowerBounds(), checkWhole) &&
               isSimplyContiguous(box, checkWhole);
      })
      .Case<fir::DeclareOp, hlfir::DeclareOp>([&](auto op) {
        return isSimplyContiguous(op.getMemref(), checkWhole);
      })
      .Case<fir::ConvertOp>(
          [&](auto op) { return isSimplyContiguous(op.getValue()); })
      .Default([](auto &&) { return false; });
}
