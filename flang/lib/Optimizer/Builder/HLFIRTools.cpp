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
#include "aiir/IR/IRMapping.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <aiir/Dialect/LLVMIR/LLVMAttrs.h>
#include <aiir/Dialect/OpenMP/OpenMPDialect.h>
#include <optional>

// Return explicit extents. If the base is a fir.box, this won't read it to
// return the extents and will instead return an empty vector.
llvm::SmallVector<aiir::Value>
hlfir::getExplicitExtentsFromShape(aiir::Value shape,
                                   fir::FirOpBuilder &builder) {
  llvm::SmallVector<aiir::Value> result;
  auto *shapeOp = shape.getDefiningOp();
  if (auto s = aiir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
    auto e = s.getExtents();
    result.append(e.begin(), e.end());
  } else if (auto s = aiir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeOp)) {
    auto e = s.getExtents();
    result.append(e.begin(), e.end());
  } else if (aiir::dyn_cast_or_null<fir::ShiftOp>(shapeOp)) {
    return {};
  } else if (auto s = aiir::dyn_cast_or_null<hlfir::ShapeOfOp>(shapeOp)) {
    hlfir::ExprType expr = aiir::cast<hlfir::ExprType>(s.getExpr().getType());
    llvm::ArrayRef<int64_t> exprShape = expr.getShape();
    aiir::Type indexTy = builder.getIndexType();
    fir::ShapeType shapeTy = aiir::cast<fir::ShapeType>(shape.getType());
    result.reserve(shapeTy.getRank());
    for (unsigned i = 0; i < shapeTy.getRank(); ++i) {
      int64_t extent = exprShape[i];
      aiir::Value extentVal;
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
static llvm::SmallVector<aiir::Value>
getExplicitExtents(fir::FortranVariableOpInterface var,
                   fir::FirOpBuilder &builder) {
  if (aiir::Value shape = var.getShape())
    return hlfir::getExplicitExtentsFromShape(var.getShape(), builder);
  return {};
}

// Return explicit lower bounds from a shape result.
// Only fir.shape, fir.shift and fir.shape_shift are currently
// supported as shape.
static llvm::SmallVector<aiir::Value>
getExplicitLboundsFromShape(aiir::Value shape) {
  llvm::SmallVector<aiir::Value> result;
  auto *shapeOp = shape.getDefiningOp();
  if (auto s = aiir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
    return {};
  } else if (auto s = aiir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeOp)) {
    auto e = s.getOrigins();
    result.append(e.begin(), e.end());
  } else if (auto s = aiir::dyn_cast_or_null<fir::ShiftOp>(shapeOp)) {
    auto e = s.getOrigins();
    result.append(e.begin(), e.end());
  } else {
    TODO(shape.getLoc(), "read fir.shape to get lower bounds");
  }
  return result;
}

// Return explicit lower bounds. For pointers and allocatables, this will not
// read the lower bounds and instead return an empty vector.
static llvm::SmallVector<aiir::Value>
getExplicitLbounds(fir::FortranVariableOpInterface var) {
  if (aiir::Value shape = var.getShape())
    return getExplicitLboundsFromShape(shape);
  return {};
}

static llvm::SmallVector<aiir::Value>
getNonDefaultLowerBounds(aiir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::Entity entity) {
  assert(!entity.isAssumedRank() &&
         "cannot compute assumed rank bounds statically");
  if (!entity.mayHaveNonDefaultLowerBounds())
    return {};
  if (auto varIface = entity.getIfVariableInterface()) {
    llvm::SmallVector<aiir::Value> lbounds = getExplicitLbounds(varIface);
    if (!lbounds.empty())
      return lbounds;
  }
  if (entity.isMutableBox())
    entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
  llvm::SmallVector<aiir::Value> lowerBounds;
  fir::factory::genDimInfoFromBox(builder, loc, entity, &lowerBounds,
                                  /*extents=*/nullptr, /*strides=*/nullptr);
  return lowerBounds;
}

static llvm::SmallVector<aiir::Value> toSmallVector(aiir::ValueRange range) {
  llvm::SmallVector<aiir::Value> res;
  res.append(range.begin(), range.end());
  return res;
}

static llvm::SmallVector<aiir::Value> getExplicitTypeParams(hlfir::Entity var) {
  if (auto varIface = var.getMaybeDereferencedVariableInterface())
    return toSmallVector(varIface.getExplicitTypeParams());
  return {};
}

static aiir::Value tryGettingNonDeferredCharLen(hlfir::Entity var) {
  if (auto varIface = var.getMaybeDereferencedVariableInterface())
    if (!varIface.getExplicitTypeParams().empty())
      return varIface.getExplicitTypeParams()[0];
  return aiir::Value{};
}

static aiir::Value genCharacterVariableLength(aiir::Location loc,
                                              fir::FirOpBuilder &builder,
                                              hlfir::Entity var) {
  if (aiir::Value len = tryGettingNonDeferredCharLen(var))
    return len;
  auto charType = aiir::cast<fir::CharacterType>(var.getFortranElementType());
  if (charType.hasConstantLen())
    return builder.createIntegerConstant(loc, builder.getIndexType(),
                                         charType.getLen());
  if (var.isMutableBox())
    var = hlfir::Entity{fir::LoadOp::create(builder, loc, var)};
  aiir::Value len = fir::factory::CharacterExprHelper{builder, loc}.getLength(
      var.getFirBase());
  assert(len && "failed to retrieve length");
  return len;
}

static fir::CharBoxValue genUnboxChar(aiir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      aiir::Value boxChar) {
  if (auto emboxChar = boxChar.getDefiningOp<fir::EmboxCharOp>())
    return {emboxChar.getMemref(), emboxChar.getLen()};
  aiir::Type refType = fir::ReferenceType::get(
      aiir::cast<fir::BoxCharType>(boxChar.getType()).getEleTy());
  auto unboxed = fir::UnboxCharOp::create(builder, loc, refType,
                                          builder.getIndexType(), boxChar);
  aiir::Value addr = unboxed.getResult(0);
  aiir::Value len = unboxed.getResult(1);
  if (auto varIface = boxChar.getDefiningOp<fir::FortranVariableOpInterface>())
    if (aiir::Value explicitlen = varIface.getExplicitCharLen())
      len = explicitlen;
  return {addr, len};
}

// To maximize chances of identifying usage of a same variables in the IR,
// always return the hlfirBase result of declare/associate if it is a raw
// pointer.
static aiir::Value getFirBaseHelper(aiir::Value hlfirBase,
                                    aiir::Value firBase) {
  if (fir::isa_ref_type(hlfirBase.getType()))
    return hlfirBase;
  return firBase;
}

aiir::Value hlfir::Entity::getFirBase() const {
  if (fir::FortranVariableOpInterface variable = getIfVariableInterface()) {
    if (auto declareOp =
            aiir::dyn_cast<hlfir::DeclareOp>(variable.getOperation()))
      return getFirBaseHelper(declareOp.getBase(), declareOp.getOriginalBase());
    if (auto associateOp =
            aiir::dyn_cast<hlfir::AssociateOp>(variable.getOperation()))
      return getFirBaseHelper(associateOp.getBase(), associateOp.getFirBase());
  }
  return getBase();
}

static bool isShapeWithLowerBounds(aiir::Value shape) {
  if (!shape)
    return false;
  auto shapeTy = shape.getType();
  return aiir::isa<fir::ShiftType>(shapeTy) ||
         aiir::isa<fir::ShapeShiftType>(shapeTy);
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

aiir::Operation *traverseConverts(aiir::Operation *op) {
  while (auto convert = llvm::dyn_cast_or_null<fir::ConvertOp>(op))
    op = convert.getValue().getDefiningOp();
  return op;
}

bool hlfir::Entity::mayBeOptional() const {
  if (!isVariable())
    return false;
  // TODO: introduce a fir type to better identify optionals.
  if (aiir::Operation *op = traverseConverts(getDefiningOp())) {
    if (auto varIface = llvm::dyn_cast<fir::FortranVariableOpInterface>(op))
      return varIface.isOptional();
    return !llvm::isa<fir::AllocaOp, fir::AllocMemOp, fir::ReboxOp,
                      fir::EmboxOp, fir::LoadOp>(op);
  }
  return true;
}

fir::FortranVariableOpInterface
hlfir::genDeclare(aiir::Location loc, fir::FirOpBuilder &builder,
                  const fir::ExtendedValue &exv, llvm::StringRef name,
                  fir::FortranVariableFlagsAttr flags, aiir::Value dummyScope,
                  aiir::Value storage, std::uint64_t storageOffset,
                  cuf::DataAttributeAttr dataAttr, unsigned dummyArgNo) {

  aiir::Value base = fir::getBase(exv);
  assert(fir::conformsWithPassByRef(base.getType()) &&
         "entity being declared must be in memory");
  aiir::Value shapeOrShift;
  llvm::SmallVector<aiir::Value> lenParams;
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
  auto declareOp = hlfir::DeclareOp::create(
      builder, loc, base, name, shapeOrShift, lenParams, dummyScope, storage,
      storageOffset, flags, dataAttr, dummyArgNo);
  return aiir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation());
}

hlfir::AssociateOp
hlfir::genAssociateExpr(aiir::Location loc, fir::FirOpBuilder &builder,
                        hlfir::Entity value, aiir::Type variableType,
                        llvm::StringRef name,
                        std::optional<aiir::NamedAttribute> attr) {
  assert(value.isValue() && "must not be a variable");
  aiir::Value shape{};
  if (value.isArray())
    shape = genShape(loc, builder, value);

  aiir::Value source = value;
  // Lowered scalar expression values for numerical and logical may have a
  // different type than what is required for the type in memory (logical
  // expressions are typically manipulated as i1, but needs to be stored
  // according to the fir.logical<kind> so that the storage size is correct).
  // Character length mismatches are ignored (it is ok for one to be dynamic
  // and the other static).
  aiir::Type varEleTy = getFortranElementType(variableType);
  aiir::Type valueEleTy = getFortranElementType(value.getType());
  if (varEleTy != valueEleTy && !(aiir::isa<fir::CharacterType>(valueEleTy) &&
                                  aiir::isa<fir::CharacterType>(varEleTy))) {
    assert(value.isScalar() && fir::isa_trivial(value.getType()));
    source = builder.createConvert(loc, fir::unwrapPassByRefType(variableType),
                                   value);
  }
  llvm::SmallVector<aiir::Value> lenParams;
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

aiir::Value hlfir::genVariableRawAddress(aiir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity var) {
  assert(var.isVariable() && "only address of variables can be taken");
  aiir::Value baseAddr = var.getFirBase();
  if (var.isMutableBox())
    baseAddr = fir::LoadOp::create(builder, loc, baseAddr);
  // Get raw address.
  if (aiir::isa<fir::BoxCharType>(var.getType()))
    baseAddr = genUnboxChar(loc, builder, var.getBase()).getAddr();
  if (aiir::isa<fir::BaseBoxType>(baseAddr.getType()))
    baseAddr = fir::BoxAddrOp::create(builder, loc, baseAddr);
  return baseAddr;
}

aiir::Value hlfir::genVariableBoxChar(aiir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::Entity var) {
  assert(var.isVariable() && "only address of variables can be taken");
  if (aiir::isa<fir::BoxCharType>(var.getType()))
    return var;
  aiir::Value addr = genVariableRawAddress(loc, builder, var);
  llvm::SmallVector<aiir::Value> lengths;
  genLengthParameters(loc, builder, var, lengths);
  assert(lengths.size() == 1);
  auto charType = aiir::cast<fir::CharacterType>(var.getFortranElementType());
  auto boxCharType =
      fir::BoxCharType::get(builder.getContext(), charType.getFKind());
  auto scalarAddr =
      builder.createConvert(loc, fir::ReferenceType::get(charType), addr);
  return fir::EmboxCharOp::create(builder, loc, boxCharType, scalarAddr,
                                  lengths[0]);
}

static hlfir::Entity changeBoxAttributes(aiir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity var,
                                         fir::BaseBoxType forceBoxType) {
  assert(llvm::isa<fir::BaseBoxType>(var.getType()) && "expect box type");
  // Propagate lower bounds.
  aiir::Value shift;
  llvm::SmallVector<aiir::Value> lbounds =
      getNonDefaultLowerBounds(loc, builder, var);
  if (!lbounds.empty())
    shift = builder.genShift(loc, lbounds);
  auto rebox = fir::ReboxOp::create(builder, loc, forceBoxType, var, shift,
                                    /*slice=*/nullptr);
  return hlfir::Entity{rebox};
}

hlfir::Entity hlfir::genVariableBox(aiir::Location loc,
                                    fir::FirOpBuilder &builder,
                                    hlfir::Entity var,
                                    fir::BaseBoxType forceBoxType) {
  assert(var.isVariable() && "must be a variable");
  var = hlfir::derefPointersAndAllocatables(loc, builder, var);
  if (aiir::isa<fir::BaseBoxType>(var.getType())) {
    if (!forceBoxType || forceBoxType == var.getType())
      return var;
    return changeBoxAttributes(loc, builder, var, forceBoxType);
  }
  // Note: if the var is not a fir.box/fir.class at that point, it has default
  // lower bounds and is not polymorphic.
  aiir::Value shape =
      var.isArray() ? hlfir::genShape(loc, builder, var) : aiir::Value{};
  llvm::SmallVector<aiir::Value> typeParams;
  aiir::Type elementType =
      forceBoxType ? fir::getFortranElementType(forceBoxType.getEleTy())
                   : var.getFortranElementType();
  auto maybeCharType = aiir::dyn_cast<fir::CharacterType>(elementType);
  if (!maybeCharType || maybeCharType.hasDynamicLen())
    hlfir::genLengthParameters(loc, builder, var, typeParams);
  aiir::Value addr = var.getBase();
  if (aiir::isa<fir::BoxCharType>(var.getType()))
    addr = genVariableRawAddress(loc, builder, var);
  const bool isVolatile = fir::isa_volatile_type(var.getType());
  aiir::Type boxType =
      fir::BoxType::get(var.getElementOrSequenceType(), isVolatile);
  if (forceBoxType) {
    boxType = forceBoxType;
    aiir::Type baseType = fir::ReferenceType::get(
        fir::unwrapRefType(forceBoxType.getEleTy()), forceBoxType.isVolatile());
    addr = builder.createConvertWithVolatileCast(loc, baseType, addr);
  }
  auto embox = fir::EmboxOp::create(builder, loc, boxType, addr, shape,
                                    /*slice=*/aiir::Value{}, typeParams);
  return hlfir::Entity{embox.getResult()};
}

hlfir::Entity hlfir::loadTrivialScalar(aiir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       Entity entity) {
  entity = derefPointersAndAllocatables(loc, builder, entity);
  if (entity.isVariable() && entity.isScalar() &&
      fir::isa_trivial(entity.getFortranElementType())) {
    // Optional entities may be represented with !fir.box<i32/f32/...>.
    // We need to take the data pointer before loading the scalar.
    aiir::Value base = genVariableRawAddress(loc, builder, entity);
    return Entity{fir::LoadOp::create(builder, loc, base)};
  }
  return entity;
}

hlfir::Entity hlfir::getElementAt(aiir::Location loc,
                                  fir::FirOpBuilder &builder, Entity entity,
                                  aiir::ValueRange oneBasedIndices) {
  if (entity.isScalar())
    return entity;
  llvm::SmallVector<aiir::Value> lenParams;
  genLengthParameters(loc, builder, entity, lenParams);
  if (aiir::isa<hlfir::ExprType>(entity.getType()))
    return hlfir::Entity{hlfir::ApplyOp::create(builder, loc, entity,
                                                oneBasedIndices, lenParams)};
  // Build hlfir.designate. The lower bounds may need to be added to
  // the oneBasedIndices since hlfir.designate expect indices
  // based on the array operand lower bounds.
  aiir::Type resultType = hlfir::getVariableElementType(entity);
  hlfir::DesignateOp designate;
  llvm::SmallVector<aiir::Value> lbounds =
      getNonDefaultLowerBounds(loc, builder, entity);
  if (!lbounds.empty()) {
    llvm::SmallVector<aiir::Value> indices;
    aiir::Type idxTy = builder.getIndexType();
    aiir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto [oneBased, lb] : llvm::zip(oneBasedIndices, lbounds)) {
      auto lbIdx = builder.createConvert(loc, idxTy, lb);
      auto oneBasedIdx = builder.createConvert(loc, idxTy, oneBased);
      auto shift = aiir::arith::SubIOp::create(builder, loc, lbIdx, one);
      aiir::Value index =
          aiir::arith::AddIOp::create(builder, loc, oneBasedIdx, shift);
      indices.push_back(index);
    }
    designate = hlfir::DesignateOp::create(builder, loc, resultType, entity,
                                           indices, lenParams);
  } else {
    designate = hlfir::DesignateOp::create(builder, loc, resultType, entity,
                                           oneBasedIndices, lenParams);
  }
  return aiir::cast<fir::FortranVariableOpInterface>(designate.getOperation());
}

static aiir::Value genUBound(aiir::Location loc, fir::FirOpBuilder &builder,
                             aiir::Value lb, aiir::Value extent,
                             aiir::Value one) {
  if (auto constantLb = fir::getIntIfConstant(lb))
    if (*constantLb == 1)
      return extent;
  extent = builder.createConvert(loc, one.getType(), extent);
  lb = builder.createConvert(loc, one.getType(), lb);
  auto add = aiir::arith::AddIOp::create(builder, loc, lb, extent);
  return aiir::arith::SubIOp::create(builder, loc, add, one);
}

llvm::SmallVector<std::pair<aiir::Value, aiir::Value>>
hlfir::genBounds(aiir::Location loc, fir::FirOpBuilder &builder,
                 Entity entity) {
  if (aiir::isa<hlfir::ExprType>(entity.getType()))
    TODO(loc, "bounds of expressions in hlfir");
  auto [exv, cleanup] = translateToExtendedValue(loc, builder, entity);
  assert(!cleanup && "translation of entity should not yield cleanup");
  if (const auto *mutableBox = exv.getBoxOf<fir::MutableBoxValue>())
    exv = fir::factory::genMutableBoxRead(builder, loc, *mutableBox);
  aiir::Type idxTy = builder.getIndexType();
  aiir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  llvm::SmallVector<std::pair<aiir::Value, aiir::Value>> result;
  for (unsigned dim = 0; dim < exv.rank(); ++dim) {
    aiir::Value extent = fir::factory::readExtent(builder, loc, exv, dim);
    aiir::Value lb = fir::factory::readLowerBound(builder, loc, exv, dim, one);
    aiir::Value ub = genUBound(loc, builder, lb, extent, one);
    result.push_back({lb, ub});
  }
  return result;
}

llvm::SmallVector<std::pair<aiir::Value, aiir::Value>>
hlfir::genBounds(aiir::Location loc, fir::FirOpBuilder &builder,
                 aiir::Value shape) {
  assert((aiir::isa<fir::ShapeShiftType>(shape.getType()) ||
          aiir::isa<fir::ShapeType>(shape.getType())) &&
         "shape must contain extents");
  auto extents = hlfir::getExplicitExtentsFromShape(shape, builder);
  auto lowers = getExplicitLboundsFromShape(shape);
  assert(lowers.empty() || lowers.size() == extents.size());
  aiir::Type idxTy = builder.getIndexType();
  aiir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  llvm::SmallVector<std::pair<aiir::Value, aiir::Value>> result;
  for (auto extent : llvm::enumerate(extents)) {
    aiir::Value lb = lowers.empty() ? one : lowers[extent.index()];
    aiir::Value ub = lowers.empty()
                         ? extent.value()
                         : genUBound(loc, builder, lb, extent.value(), one);
    result.push_back({lb, ub});
  }
  return result;
}

llvm::SmallVector<aiir::Value> hlfir::genLowerbounds(aiir::Location loc,
                                                     fir::FirOpBuilder &builder,
                                                     aiir::Value shape,
                                                     unsigned rank) {
  llvm::SmallVector<aiir::Value> lbounds;
  if (shape)
    lbounds = getExplicitLboundsFromShape(shape);
  if (!lbounds.empty())
    return lbounds;
  aiir::Value one =
      builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  return llvm::SmallVector<aiir::Value>(rank, one);
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

static aiir::Value computeVariableExtent(aiir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity variable,
                                         fir::SequenceType seqTy,
                                         unsigned dim) {
  aiir::Type idxTy = builder.getIndexType();
  if (seqTy.getShape().size() > dim) {
    fir::SequenceType::Extent typeExtent = seqTy.getShape()[dim];
    if (typeExtent != fir::SequenceType::getUnknownExtent())
      return builder.createIntegerConstant(loc, idxTy, typeExtent);
  }
  assert(aiir::isa<fir::BaseBoxType>(variable.getType()) &&
         "array variable with dynamic extent must be boxed");
  aiir::Value dimVal = builder.createIntegerConstant(loc, idxTy, dim);
  auto dimInfo = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy,
                                        variable, dimVal);
  return dimInfo.getExtent();
}
llvm::SmallVector<aiir::Value> getVariableExtents(aiir::Location loc,
                                                  fir::FirOpBuilder &builder,
                                                  hlfir::Entity variable) {
  llvm::SmallVector<aiir::Value> extents;
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
  fir::SequenceType seqTy = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(variable.getType()));
  unsigned rank = seqTy.getShape().size();
  for (unsigned dim = 0; dim < rank; ++dim)
    extents.push_back(
        computeVariableExtent(loc, builder, variable, seqTy, dim));
  return extents;
}

static aiir::Value tryRetrievingShapeOrShift(hlfir::Entity entity) {
  if (aiir::isa<hlfir::ExprType>(entity.getType())) {
    if (auto elemental = entity.getDefiningOp<hlfir::ElementalOp>())
      return elemental.getShape();
    if (auto evalInMem = entity.getDefiningOp<hlfir::EvaluateInMemoryOp>())
      return evalInMem.getShape();
    return aiir::Value{};
  }
  if (auto varIface = entity.getIfVariableInterface())
    return varIface.getShape();
  return {};
}

aiir::Value hlfir::genShape(aiir::Location loc, fir::FirOpBuilder &builder,
                            hlfir::Entity entity) {
  assert(entity.isArray() && "entity must be an array");
  entity = followShapeInducingSource(entity);
  assert(entity && "what?");
  if (auto shape = tryRetrievingShapeOrShift(entity)) {
    if (aiir::isa<fir::ShapeType>(shape.getType()))
      return shape;
    if (aiir::isa<fir::ShapeShiftType>(shape.getType()))
      if (auto s = shape.getDefiningOp<fir::ShapeShiftOp>())
        return fir::ShapeOp::create(builder, loc, s.getExtents());
  }
  if (aiir::isa<hlfir::ExprType>(entity.getType()))
    return hlfir::ShapeOfOp::create(builder, loc, entity.getBase());
  // There is no shape lying around for this entity. Retrieve the extents and
  // build a new fir.shape.
  return fir::ShapeOp::create(builder, loc,
                              getVariableExtents(loc, builder, entity));
}

llvm::SmallVector<aiir::Value>
hlfir::getIndexExtents(aiir::Location loc, fir::FirOpBuilder &builder,
                       aiir::Value shape) {
  llvm::SmallVector<aiir::Value> extents =
      hlfir::getExplicitExtentsFromShape(shape, builder);
  aiir::Type indexType = builder.getIndexType();
  for (auto &extent : extents)
    extent = builder.createConvert(loc, indexType, extent);
  return extents;
}

aiir::Value hlfir::genExtent(aiir::Location loc, fir::FirOpBuilder &builder,
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
    fir::SequenceType seqTy = aiir::cast<fir::SequenceType>(
        hlfir::getFortranElementOrSequenceType(entity.getType()));
    return computeVariableExtent(loc, builder, entity, seqTy, dim);
  }
  TODO(loc, "get extent from HLFIR expr without producer holding the shape");
}

aiir::Value hlfir::genLBound(aiir::Location loc, fir::FirOpBuilder &builder,
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
  assert(aiir::isa<fir::BaseBoxType>(entity.getType()) && "must be a box");
  aiir::Type idxTy = builder.getIndexType();
  aiir::Value dimVal = builder.createIntegerConstant(loc, idxTy, dim);
  auto dimInfo =
      fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, entity, dimVal);
  return dimInfo.getLowerBound();
}

static bool
getExprLengthParameters(aiir::Value expr,
                        llvm::SmallVectorImpl<aiir::Value> &result) {
  if (auto concat = expr.getDefiningOp<hlfir::ConcatOp>()) {
    result.push_back(concat.getLength());
    return true;
  }
  if (auto setLen = expr.getDefiningOp<hlfir::SetLengthOp>()) {
    result.push_back(setLen.getLength());
    return true;
  }
  if (auto elemental = expr.getDefiningOp<hlfir::ElementalOp>()) {
    result.append(elemental.getTypeparams().begin(),
                  elemental.getTypeparams().end());
    return true;
  }
  if (auto evalInMem = expr.getDefiningOp<hlfir::EvaluateInMemoryOp>()) {
    result.append(evalInMem.getTypeparams().begin(),
                  evalInMem.getTypeparams().end());
    return true;
  }
  if (auto apply = expr.getDefiningOp<hlfir::ApplyOp>()) {
    result.append(apply.getTypeparams().begin(), apply.getTypeparams().end());
    return true;
  }
  return false;
}

void hlfir::genLengthParameters(aiir::Location loc, fir::FirOpBuilder &builder,
                                Entity entity,
                                llvm::SmallVectorImpl<aiir::Value> &result) {
  if (!entity.hasLengthParameters())
    return;
  if (aiir::isa<hlfir::ExprType>(entity.getType())) {
    aiir::Value expr = entity;
    if (auto reassoc = expr.getDefiningOp<hlfir::NoReassocOp>())
      expr = reassoc.getVal();
    // Going through fir::ExtendedValue would create a temp,
    // which is not desired for an inquiry.
    // TODO: make this an interface when adding further character producing ops.

    if (auto asExpr = expr.getDefiningOp<hlfir::AsExprOp>()) {
      hlfir::genLengthParameters(loc, builder, hlfir::Entity{asExpr.getVar()},
                                 result);
      return;
    }
    if (getExprLengthParameters(expr, result))
      return;
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

aiir::Value hlfir::genCharLength(aiir::Location loc, fir::FirOpBuilder &builder,
                                 hlfir::Entity entity) {
  llvm::SmallVector<aiir::Value, 1> lenParams;
  genLengthParameters(loc, builder, entity, lenParams);
  assert(lenParams.size() == 1 && "characters must have one length parameters");
  return lenParams[0];
}

std::optional<std::int64_t> hlfir::getCharLengthIfConst(hlfir::Entity entity) {
  if (!entity.isCharacter()) {
    return std::nullopt;
  }
  if (aiir::isa<hlfir::ExprType>(entity.getType())) {
    aiir::Value expr = entity;
    if (auto reassoc = expr.getDefiningOp<hlfir::NoReassocOp>())
      expr = reassoc.getVal();

    if (auto asExpr = expr.getDefiningOp<hlfir::AsExprOp>())
      return getCharLengthIfConst(hlfir::Entity{asExpr.getVar()});

    llvm::SmallVector<aiir::Value> param;
    if (getExprLengthParameters(expr, param)) {
      assert(param.size() == 1 && "characters must have one length parameters");
      return fir::getIntIfConstant(param.pop_back_val());
    }
    return std::nullopt;
  }

  // entity is a var
  if (aiir::Value len = tryGettingNonDeferredCharLen(entity))
    return fir::getIntIfConstant(len);
  auto charType =
      aiir::cast<fir::CharacterType>(entity.getFortranElementType());
  if (charType.hasConstantLen())
    return charType.getLen();
  return std::nullopt;
}

aiir::Value hlfir::genRank(aiir::Location loc, fir::FirOpBuilder &builder,
                           hlfir::Entity entity, aiir::Type resultType) {
  if (!entity.isAssumedRank())
    return builder.createIntegerConstant(loc, resultType, entity.getRank());
  assert(entity.isBoxAddressOrValue() &&
         "assumed-ranks are box addresses or values");
  return fir::BoxRankOp::create(builder, loc, resultType, entity);
}

// Return a "shape" that can be used in fir.embox/fir.rebox with \p exv base.
static aiir::Value asEmboxShape(aiir::Location loc, fir::FirOpBuilder &builder,
                                const fir::ExtendedValue &exv,
                                aiir::Value shape) {
  if (!shape)
    return shape;
  // fir.rebox does not need and does not accept extents (fir.shape or
  // fir.shape_shift) since this information is already in the input fir.box,
  // it only accepts fir.shift because local lower bounds may not be reflected
  // in the fir.box.
  if (aiir::isa<fir::BaseBoxType>(fir::getBase(exv).getType()) &&
      !aiir::isa<fir::ShiftType>(shape.getType()))
    return builder.createShape(loc, exv);
  return shape;
}

std::pair<aiir::Value, aiir::Value> hlfir::genVariableFirBaseShapeAndParams(
    aiir::Location loc, fir::FirOpBuilder &builder, Entity entity,
    llvm::SmallVectorImpl<aiir::Value> &typeParams) {
  auto [exv, cleanup] = translateToExtendedValue(loc, builder, entity);
  assert(!cleanup && "variable to Exv should not produce cleanup");
  if (entity.hasLengthParameters()) {
    auto params = fir::getTypeParams(exv);
    typeParams.append(params.begin(), params.end());
  }
  if (entity.isScalar())
    return {fir::getBase(exv), aiir::Value{}};

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
    aiir::Value shape = variableInterface.getShape();
    if (aiir::isa<fir::BaseBoxType>(fir::getBase(exv).getType()) ||
        !aiir::isa<fir::BaseBoxType>(entity.getType()) ||
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

hlfir::Entity hlfir::derefPointersAndAllocatables(aiir::Location loc,
                                                  fir::FirOpBuilder &builder,
                                                  Entity entity) {
  if (entity.isMutableBox()) {
    hlfir::Entity boxLoad{fir::LoadOp::create(builder, loc, entity)};
    if (entity.isScalar()) {
      if (!entity.isPolymorphic() && !entity.hasLengthParameters())
        return hlfir::Entity{fir::BoxAddrOp::create(builder, loc, boxLoad)};
      aiir::Type elementType = boxLoad.getFortranElementType();
      if (auto charType = aiir::dyn_cast<fir::CharacterType>(elementType)) {
        aiir::Value base = fir::BoxAddrOp::create(builder, loc, boxLoad);
        if (charType.hasConstantLen())
          return hlfir::Entity{base};
        aiir::Value len = genCharacterVariableLength(loc, builder, entity);
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

aiir::Type hlfir::getVariableElementType(hlfir::Entity variable) {
  assert(variable.isVariable() && "entity must be a variable");
  if (variable.isScalar())
    return variable.getType();
  aiir::Type eleTy = variable.getFortranElementType();
  const bool isVolatile = fir::isa_volatile_type(variable.getType());
  if (variable.isPolymorphic())
    return fir::ClassType::get(eleTy, isVolatile);
  if (auto charType = aiir::dyn_cast<fir::CharacterType>(eleTy)) {
    if (charType.hasDynamicLen())
      return fir::BoxCharType::get(charType.getContext(), charType.getFKind());
  } else if (fir::isRecordWithTypeParameters(eleTy)) {
    return fir::BoxType::get(eleTy, isVolatile);
  }
  return fir::ReferenceType::get(eleTy, isVolatile);
}

aiir::Type hlfir::getEntityElementType(hlfir::Entity entity) {
  if (entity.isVariable())
    return getVariableElementType(entity);
  if (entity.isScalar())
    return entity.getType();
  auto exprType = aiir::dyn_cast<hlfir::ExprType>(entity.getType());
  assert(exprType && "array value must be an hlfir.expr");
  return exprType.getElementExprType();
}

static hlfir::ExprType getArrayExprType(aiir::Type elementType,
                                        aiir::Value shape, bool isPolymorphic) {
  unsigned rank = aiir::cast<fir::ShapeType>(shape.getType()).getRank();
  hlfir::ExprType::Shape typeShape(rank, hlfir::ExprType::getUnknownExtent());
  if (auto shapeOp = shape.getDefiningOp<fir::ShapeOp>())
    for (auto extent : llvm::enumerate(shapeOp.getExtents()))
      if (auto cstExtent = fir::getIntIfConstant(extent.value()))
        typeShape[extent.index()] = *cstExtent;
  return hlfir::ExprType::get(elementType.getContext(), typeShape, elementType,
                              isPolymorphic);
}

hlfir::ElementalOp hlfir::genElementalOp(
    aiir::Location loc, fir::FirOpBuilder &builder, aiir::Type elementType,
    aiir::Value shape, aiir::ValueRange typeParams,
    const ElementalKernelGenerator &genKernel, bool isUnordered,
    aiir::Value polymorphicMold, aiir::Type exprType) {
  if (!exprType)
    exprType = getArrayExprType(elementType, shape, !!polymorphicMold);
  auto elementalOp = hlfir::ElementalOp::create(
      builder, loc, exprType, shape, polymorphicMold, typeParams, isUnordered);
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(elementalOp.getBody());
  aiir::Value elementResult = genKernel(loc, builder, elementalOp.getIndices());
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
hlfir::inlineElementalOp(aiir::Location loc, fir::FirOpBuilder &builder,
                         hlfir::ElementalOp elemental,
                         aiir::ValueRange oneBasedIndices) {
  // hlfir.elemental region is a SizedRegion<1>.
  assert(elemental.getRegion().hasOneBlock() &&
         "expect elemental region to have one block");
  aiir::IRMapping mapper;
  mapper.map(elemental.getIndices(), oneBasedIndices);
  aiir::Operation *newOp;
  for (auto &op : elemental.getRegion().back().getOperations())
    newOp = builder.clone(op, mapper);
  auto yield = aiir::dyn_cast_or_null<hlfir::YieldElementOp>(newOp);
  assert(yield && "last ElementalOp operation must be am hlfir.yield_element");
  return yield;
}

aiir::Value hlfir::inlineElementalOp(
    aiir::Location loc, fir::FirOpBuilder &builder,
    hlfir::ElementalOpInterface elemental, aiir::ValueRange oneBasedIndices,
    aiir::IRMapping &mapper,
    const std::function<bool(hlfir::ElementalOp)> &mustRecursivelyInline) {
  aiir::Region &region = elemental.getElementalRegion();
  // hlfir.elemental region is a SizedRegion<1>.
  assert(region.hasOneBlock() && "elemental region must have one block");
  mapper.map(elemental.getIndices(), oneBasedIndices);
  for (auto &op : region.front().without_terminator()) {
    if (auto apply = aiir::dyn_cast<hlfir::ApplyOp>(op))
      if (auto appliedElemental =
              apply.getExpr().getDefiningOp<hlfir::ElementalOp>())
        if (mustRecursivelyInline(appliedElemental)) {
          llvm::SmallVector<aiir::Value> clonedApplyIndices;
          for (auto indice : apply.getIndices())
            clonedApplyIndices.push_back(mapper.lookupOrDefault(indice));
          hlfir::ElementalOpInterface elementalIface =
              aiir::cast<hlfir::ElementalOpInterface>(
                  appliedElemental.getOperation());
          aiir::Value inlined = inlineElementalOp(loc, builder, elementalIface,
                                                  clonedApplyIndices, mapper,
                                                  mustRecursivelyInline);
          mapper.map(apply.getResult(), inlined);
          continue;
        }
    (void)builder.clone(op, mapper);
  }
  return mapper.lookupOrDefault(elemental.getElementEntity());
}

hlfir::LoopNest hlfir::genLoopNest(aiir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   aiir::ValueRange extents, bool isUnordered,
                                   bool emitWorkshareLoop,
                                   bool couldVectorize) {
  emitWorkshareLoop = emitWorkshareLoop && isUnordered;
  hlfir::LoopNest loopNest;
  assert(!extents.empty() && "must have at least one extent");
  aiir::OpBuilder::InsertionGuard guard(builder);
  loopNest.oneBasedIndices.assign(extents.size(), aiir::Value{});
  // Build loop nest from column to row.
  auto one = aiir::arith::ConstantIndexOp::create(builder, loc, 1);
  aiir::Type indexType = builder.getIndexType();
  if (emitWorkshareLoop) {
    auto wslw = aiir::omp::WorkshareLoopWrapperOp::create(builder, loc);
    loopNest.outerOp = wslw;
    builder.createBlock(&wslw.getRegion());
    aiir::omp::LoopNestOperands lnops;
    lnops.loopInclusive = builder.getUnitAttr();
    for (auto extent : llvm::reverse(extents)) {
      lnops.loopLowerBounds.push_back(one);
      lnops.loopUpperBounds.push_back(extent);
      lnops.loopSteps.push_back(one);
    }
    auto lnOp = aiir::omp::LoopNestOp::create(builder, loc, lnops);
    aiir::Block *block = builder.createBlock(&lnOp.getRegion());
    for (auto extent : llvm::reverse(extents))
      block->addArgument(extent.getType(), extent.getLoc());
    loopNest.body = block;
    aiir::omp::YieldOp::create(builder, loc);
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
        aiir::LLVM::LoopVectorizeAttr va{aiir::LLVM::LoopVectorizeAttr::get(
            builder.getContext(),
            /*disable=*/builder.getBoolAttr(true), {}, {}, {}, {}, {}, {})};
        aiir::LLVM::LoopAnnotationAttr la = aiir::LLVM::LoopAnnotationAttr::get(
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

llvm::SmallVector<aiir::Value> hlfir::genLoopNestWithReductions(
    aiir::Location loc, fir::FirOpBuilder &builder, aiir::ValueRange extents,
    aiir::ValueRange reductionInits, const ReductionLoopBodyGenerator &genBody,
    bool isUnordered) {
  assert(!extents.empty() && "must have at least one extent");
  // Build loop nest from column to row.
  auto one = aiir::arith::ConstantIndexOp::create(builder, loc, 1);
  aiir::Type indexType = builder.getIndexType();
  unsigned dim = extents.size() - 1;
  fir::DoLoopOp outerLoop = nullptr;
  fir::DoLoopOp parentLoop = nullptr;
  llvm::SmallVector<aiir::Value> oneBasedIndices;
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

  llvm::SmallVector<aiir::Value> reductionValues;
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
conditionallyEvaluate(aiir::Location loc, fir::FirOpBuilder &builder,
                      aiir::Value condition, const Lambda &genIfTrue) {
  aiir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();

  // Evaluate in some region that will be moved into the actual ifOp (the actual
  // ifOp can only be created when the result types are known).
  auto badIfOp = fir::IfOp::create(builder, loc, condition.getType(), condition,
                                   /*withElseRegion=*/false);
  aiir::Block *preparationBlock = &badIfOp.getThenRegion().front();
  builder.setInsertionPointToStart(preparationBlock);
  fir::ExtendedValue result = genIfTrue();
  fir::ResultOp resultOp = result.match(
      [&](const fir::CharBoxValue &box) -> fir::ResultOp {
        return fir::ResultOp::create(
            builder, loc, aiir::ValueRange{box.getAddr(), box.getLen()});
      },
      [&](const aiir::Value &addr) -> fir::ResultOp {
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
  llvm::SmallVector<aiir::Value> absentValues;
  for (aiir::Type resTy : ifOp->getResultTypes()) {
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
      [&](const aiir::Value &) -> fir::ExtendedValue {
        return ifOp.getResult(0);
      },
      [&](const auto &) -> fir::ExtendedValue {
        TODO(loc, "unboxing non scalar optional fir.box");
      });
}

static fir::ExtendedValue translateVariableToExtendedValue(
    aiir::Location loc, fir::FirOpBuilder &builder, hlfir::Entity variable,
    bool forceHlfirBase = false, bool contiguousHint = false,
    bool keepScalarOptionalBoxed = false) {
  assert(variable.isVariable() && "must be a variable");
  // When going towards FIR, use the original base value to avoid
  // introducing descriptors at runtime when they are not required.
  // This is not done for assumed-rank since the fir::ExtendedValue cannot
  // held the related lower bounds in an vector. The lower bounds of the
  // descriptor must always be used instead.

  aiir::Value base = (forceHlfirBase || variable.isAssumedRank())
                         ? variable.getBase()
                         : variable.getFirBase();
  if (variable.isMutableBox())
    return fir::MutableBoxValue(base, getExplicitTypeParams(variable),
                                fir::MutableProperties{});

  if (aiir::isa<fir::BaseBoxType>(base.getType())) {
    const bool contiguous = variable.isSimplyContiguous() || contiguousHint;
    const bool isAssumedRank = variable.isAssumedRank();
    if (!contiguous || variable.isPolymorphic() ||
        variable.isDerivedWithLengthParameters() || isAssumedRank) {
      llvm::SmallVector<aiir::Value> nonDefaultLbounds;
      if (!isAssumedRank)
        nonDefaultLbounds = getNonDefaultLowerBounds(loc, builder, variable);
      return fir::BoxValue(base, nonDefaultLbounds,
                           getExplicitTypeParams(variable));
    }
    if (variable.mayBeOptional()) {
      if (!keepScalarOptionalBoxed && variable.isScalar()) {
        aiir::Value isPresent = fir::IsPresentOp::create(
            builder, loc, builder.getI1Type(), variable);
        return conditionallyEvaluate(
            loc, builder, isPresent, [&]() -> fir::ExtendedValue {
              aiir::Value base = genVariableRawAddress(loc, builder, variable);
              if (variable.isCharacter()) {
                aiir::Value len =
                    genCharacterVariableLength(loc, builder, variable);
                return fir::CharBoxValue{base, len};
              }
              return base;
            });
      }
      llvm::SmallVector<aiir::Value> nonDefaultLbounds =
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
      if (aiir::isa<fir::BoxCharType>(base.getType()))
        return genUnboxChar(loc, builder, base);
      aiir::Value len = genCharacterVariableLength(loc, builder, variable);
      return fir::CharBoxValue{base, len};
    }
    return base;
  }
  llvm::SmallVector<aiir::Value> extents;
  llvm::SmallVector<aiir::Value> nonDefaultLbounds;
  if (aiir::isa<fir::BaseBoxType>(variable.getType()) &&
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
hlfir::translateToExtendedValue(aiir::Location loc, fir::FirOpBuilder &builder,
                                fir::FortranVariableOpInterface var,
                                bool forceHlfirBase) {
  return translateVariableToExtendedValue(loc, builder, var, forceHlfirBase);
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::translateToExtendedValue(aiir::Location loc, fir::FirOpBuilder &builder,
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
    return {static_cast<aiir::Value>(entity), std::nullopt};
  }

  if (aiir::isa<hlfir::ExprType>(entity.getType())) {
    aiir::NamedAttribute byRefAttr = fir::getAdaptToByRefAttr(builder);
    hlfir::AssociateOp associate = hlfir::genAssociateExpr(
        loc, builder, entity, entity.getType(), "", byRefAttr);
    auto *bldr = &builder;
    hlfir::CleanupFunction cleanup = [bldr, loc, associate]() -> void {
      hlfir::EndAssociateOp::create(*bldr, loc, associate);
    };
    hlfir::Entity temp{associate.getBase()};
    return {translateToExtendedValue(loc, builder, temp).first, cleanup};
  }
  return {{static_cast<aiir::Value>(entity)}, {}};
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::convertToValue(aiir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity entity) {
  // Load scalar references to integer, logical, real, or complex value
  // to an aiir value, dereference allocatable and pointers, and get rid
  // of fir.box that are not needed or create a copy into contiguous memory.
  auto derefedAndLoadedEntity = loadTrivialScalar(loc, builder, entity);
  return translateToExtendedValue(loc, builder, derefedAndLoadedEntity);
}

static fir::ExtendedValue placeTrivialInMemory(aiir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               aiir::Value val,
                                               aiir::Type targetType) {
  auto temp = builder.createTemporary(loc, targetType);
  if (targetType != val.getType())
    builder.createStoreWithConvert(loc, val, temp);
  else
    fir::StoreOp::create(builder, loc, val, temp);
  return temp;
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::convertToBox(aiir::Location loc, fir::FirOpBuilder &builder,
                    hlfir::Entity entity, aiir::Type targetType) {
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
  aiir::Value base = fir::getBase(exv);
  if (fir::isa_trivial(base.getType()))
    exv = placeTrivialInMemory(loc, builder, base, targetType);
  fir::BoxValue box = fir::factory::createBoxValue(builder, loc, exv);
  return {box, cleanup};
}

std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>
hlfir::convertToAddress(aiir::Location loc, fir::FirOpBuilder &builder,
                        hlfir::Entity entity, aiir::Type targetType) {
  hlfir::Entity derefedEntity =
      hlfir::derefPointersAndAllocatables(loc, builder, entity);
  auto [exv, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, derefedEntity);
  aiir::Value base = fir::getBase(exv);
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
hlfir::cloneToElementalOp(aiir::Location loc, fir::FirOpBuilder &builder,
                          hlfir::ElementalAddrOp elementalAddrOp) {
  hlfir::Entity scalarAddress =
      hlfir::Entity{aiir::cast<hlfir::YieldOp>(
                        elementalAddrOp.getBody().back().getTerminator())
                        .getEntity()};
  llvm::SmallVector<aiir::Value, 1> typeParams;
  hlfir::genLengthParameters(loc, builder, scalarAddress, typeParams);

  builder.setInsertionPointAfter(elementalAddrOp);
  auto genKernel = [&](aiir::Location l, fir::FirOpBuilder &b,
                       aiir::ValueRange oneBasedIndices) -> hlfir::Entity {
    aiir::IRMapping mapper;
    mapper.map(elementalAddrOp.getIndices(), oneBasedIndices);
    aiir::Operation *newOp = nullptr;
    for (auto &op : elementalAddrOp.getBody().back().getOperations())
      newOp = b.clone(op, mapper);
    auto newYielOp = aiir::dyn_cast_or_null<hlfir::YieldOp>(newOp);
    assert(newYielOp && "hlfir.elemental_addr is ill formed");
    hlfir::Entity newAddr{newYielOp.getEntity()};
    newYielOp->erase();
    return hlfir::loadTrivialScalar(l, b, newAddr);
  };
  aiir::Type elementType = scalarAddress.getFortranElementType();
  return hlfir::genElementalOp(
      loc, builder, elementType, elementalAddrOp.getShape(), typeParams,
      genKernel, !elementalAddrOp.isOrdered(), elementalAddrOp.getMold());
}

bool hlfir::elementalOpMustProduceTemp(hlfir::ElementalOp elemental) {
  for (aiir::Operation *useOp : elemental->getUsers())
    if (auto destroy = aiir::dyn_cast<hlfir::DestroyOp>(useOp))
      if (destroy.mustFinalizeExpr())
        return true;

  return false;
}

static void combineAndStoreElement(
    aiir::Location loc, fir::FirOpBuilder &builder, hlfir::Entity lhs,
    hlfir::Entity rhs, bool temporaryLHS,
    std::function<void(aiir::Location, fir::FirOpBuilder &, hlfir::Entity,
                       hlfir::Entity, aiir::ArrayAttr)> *scalarCombineAndAssign,
    aiir::ArrayAttr accessGroups) {
  if (scalarCombineAndAssign) {
    (*scalarCombineAndAssign)(loc, builder, lhs, rhs, accessGroups);
    return;
  }
  hlfir::Entity valueToAssign = hlfir::loadTrivialScalar(loc, builder, rhs);
  if (accessGroups)
    if (auto load = valueToAssign.getDefiningOp<fir::LoadOp>())
      load.setAccessGroupsAttr(accessGroups);
  auto assign = hlfir::AssignOp::create(builder, loc, valueToAssign, lhs,
                                        /*realloc=*/false,
                                        /*keep_lhs_length_if_realloc=*/false,
                                        /*temporary_lhs=*/temporaryLHS);
  if (accessGroups)
    assign->setAttr(fir::getAccessGroupsAttrName(), accessGroups);
}

void hlfir::genNoAliasArrayAssignment(
    aiir::Location loc, fir::FirOpBuilder &builder, hlfir::Entity rhs,
    hlfir::Entity lhs, bool emitWorkshareLoop, bool temporaryLHS,
    std::function<void(aiir::Location, fir::FirOpBuilder &, hlfir::Entity,
                       hlfir::Entity, aiir::ArrayAttr)> *scalarCombineAndAssign,
    aiir::ArrayAttr accessGroups) {
  aiir::OpBuilder::InsertionGuard guard(builder);
  rhs = hlfir::derefPointersAndAllocatables(loc, builder, rhs);
  lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
  aiir::Value lhsShape = hlfir::genShape(loc, builder, lhs);
  llvm::SmallVector<aiir::Value> extents =
      hlfir::getIndexExtents(loc, builder, lhsShape);
  if (rhs.isArray()) {
    aiir::Value rhsShape = hlfir::genShape(loc, builder, rhs);
    llvm::SmallVector<aiir::Value> rhsExtents =
        hlfir::getIndexExtents(loc, builder, rhsShape);
    extents = fir::factory::deduceOptimalExtents(extents, rhsExtents);
  }
  hlfir::LoopNest loopNest =
      hlfir::genLoopNest(loc, builder, extents,
                         /*isUnordered=*/true, emitWorkshareLoop);
  builder.setInsertionPointToStart(loopNest.body);
  auto rhsArrayElement =
      hlfir::getElementAt(loc, builder, rhs, loopNest.oneBasedIndices);
  if (!scalarCombineAndAssign)
    rhsArrayElement = hlfir::loadTrivialScalar(loc, builder, rhsArrayElement);
  auto lhsArrayElement =
      hlfir::getElementAt(loc, builder, lhs, loopNest.oneBasedIndices);
  combineAndStoreElement(loc, builder, lhsArrayElement, rhsArrayElement,
                         temporaryLHS, scalarCombineAndAssign, accessGroups);
}

void hlfir::genNoAliasAssignment(
    aiir::Location loc, fir::FirOpBuilder &builder, hlfir::Entity rhs,
    hlfir::Entity lhs, bool emitWorkshareLoop, bool temporaryLHS,
    std::function<void(aiir::Location, fir::FirOpBuilder &, hlfir::Entity,
                       hlfir::Entity, aiir::ArrayAttr)> *scalarCombineAndAssign,
    aiir::ArrayAttr accessGroups) {
  if (lhs.isArray()) {
    genNoAliasArrayAssignment(loc, builder, rhs, lhs, emitWorkshareLoop,
                              temporaryLHS, scalarCombineAndAssign,
                              accessGroups);
    return;
  }
  rhs = hlfir::derefPointersAndAllocatables(loc, builder, rhs);
  lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
  combineAndStoreElement(loc, builder, lhs, rhs, temporaryLHS,
                         scalarCombineAndAssign, accessGroups);
}

std::pair<hlfir::Entity, bool>
hlfir::createTempFromMold(aiir::Location loc, fir::FirOpBuilder &builder,
                          hlfir::Entity mold) {
  assert(!mold.isAssumedRank() &&
         "cannot create temporary from assumed-rank mold");
  llvm::SmallVector<aiir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};

  aiir::Value shape{};
  llvm::SmallVector<aiir::Value> extents;
  if (mold.isArray()) {
    shape = hlfir::genShape(loc, builder, mold);
    extents = hlfir::getExplicitExtentsFromShape(shape, builder);
  }

  bool useStack = !mold.isArray() && !mold.isPolymorphic();
  auto genTempDeclareOp =
      [](fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value memref,
         llvm::StringRef name, aiir::Value shape,
         llvm::ArrayRef<aiir::Value> typeParams,
         fir::FortranVariableFlagsAttr attrs) -> aiir::Value {
    auto declareOp =
        hlfir::DeclareOp::create(builder, loc, memref, name, shape, typeParams,
                                 /*dummy_scope=*/nullptr, /*storage=*/nullptr,
                                 /*storage_offset=*/0, attrs);
    return declareOp.getBase();
  };

  auto [base, isHeapAlloc] = builder.createAndDeclareTemp(
      loc, mold.getElementOrSequenceType(), shape, extents, lenParams,
      genTempDeclareOp, mold.isPolymorphic() ? mold.getBase() : nullptr,
      useStack, tmpName);
  return {hlfir::Entity{base}, isHeapAlloc};
}

hlfir::Entity hlfir::createStackTempFromMold(aiir::Location loc,
                                             fir::FirOpBuilder &builder,
                                             hlfir::Entity mold) {
  llvm::SmallVector<aiir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};
  aiir::Value alloc;
  aiir::Value shape{};
  fir::FortranVariableFlagsAttr declAttrs;

  if (mold.isPolymorphic()) {
    // genAllocatableApplyMold does heap allocation
    TODO(loc, "createStackTempFromMold for polymorphic type");
  } else if (mold.isArray()) {
    aiir::Type sequenceType =
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
                               /*dummy_scope=*/nullptr, /*storage=*/nullptr,
                               /*storage_offset=*/0, declAttrs);
  return hlfir::Entity{declareOp.getBase()};
}

hlfir::EntityWithAttributes
hlfir::convertCharacterKind(aiir::Location loc, fir::FirOpBuilder &builder,
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
      /*typeparams=*/aiir::ValueRange{res.getLen()})};
}

std::pair<hlfir::Entity, std::optional<hlfir::CleanupFunction>>
hlfir::genTypeAndKindConvert(aiir::Location loc, fir::FirOpBuilder &builder,
                             hlfir::Entity source, aiir::Type toType,
                             bool preserveLowerBounds) {
  aiir::Type fromType = source.getFortranElementType();
  toType = hlfir::getFortranElementType(toType);
  if (!toType || fromType == toType ||
      !(fir::isa_trivial(toType) || aiir::isa<fir::CharacterType>(toType)))
    return {source, std::nullopt};

  std::optional<int> toKindCharConvert;
  if (auto toCharTy = aiir::dyn_cast<fir::CharacterType>(toType)) {
    if (auto fromCharTy = aiir::dyn_cast<fir::CharacterType>(fromType))
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
    aiir::Value cast = toKindCharConvert
                           ? aiir::Value{hlfir::convertCharacterKind(
                                 loc, builder, source, *toKindCharConvert)}
                           : builder.convertWithSemantics(loc, toType, source);
    return {hlfir::Entity{cast}, std::nullopt};
  }

  aiir::Value shape = hlfir::genShape(loc, builder, source);
  auto genKernel = [source, toType, toKindCharConvert](
                       aiir::Location loc, fir::FirOpBuilder &builder,
                       aiir::ValueRange oneBasedIndices) -> hlfir::Entity {
    auto elementPtr =
        hlfir::getElementAt(loc, builder, source, oneBasedIndices);
    auto val = hlfir::loadTrivialScalar(loc, builder, elementPtr);
    if (toKindCharConvert)
      return hlfir::convertCharacterKind(loc, builder, val, *toKindCharConvert);
    return hlfir::EntityWithAttributes{
        builder.convertWithSemantics(loc, toType, val)};
  };
  llvm::SmallVector<aiir::Value, 1> lenParams;
  hlfir::genLengthParameters(loc, builder, source, lenParams);
  aiir::Value convertedRhs =
      hlfir::genElementalOp(loc, builder, toType, shape, lenParams, genKernel,
                            /*isUnordered=*/true);

  if (preserveLowerBounds && source.mayHaveNonDefaultLowerBounds()) {
    hlfir::AssociateOp associate =
        genAssociateExpr(loc, builder, hlfir::Entity{convertedRhs},
                         convertedRhs.getType(), ".tmp.keeplbounds");
    fir::ShapeOp shapeOp = associate.getShape().getDefiningOp<fir::ShapeOp>();
    assert(shapeOp && "associate shape must be a fir.shape");
    const unsigned rank = shapeOp.getExtents().size();
    llvm::SmallVector<aiir::Value> lbAndExtents;
    for (unsigned dim = 0; dim < rank; ++dim) {
      lbAndExtents.push_back(hlfir::genLBound(loc, builder, source, dim));
      lbAndExtents.push_back(shapeOp.getExtents()[dim]);
    }
    auto shapeShiftType = fir::ShapeShiftType::get(builder.getContext(), rank);
    aiir::Value shapeShift =
        fir::ShapeShiftOp::create(builder, loc, shapeShiftType, lbAndExtents);
    auto declareOp = hlfir::DeclareOp::create(
        builder, loc, associate.getFirBase(), *associate.getUniqName(),
        shapeShift, associate.getTypeparams());
    hlfir::Entity castWithLbounds =
        aiir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation());
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
    aiir::Location loc, fir::FirOpBuilder &builder,
    hlfir::EvaluateInMemoryOp evalInMem, aiir::Value shape,
    aiir::ValueRange typeParams) {
  llvm::StringRef tmpName{".tmp.expr_result"};
  llvm::SmallVector<aiir::Value> extents =
      hlfir::getIndexExtents(loc, builder, shape);
  aiir::Type baseType =
      hlfir::getFortranElementOrSequenceType(evalInMem.getType());
  bool heapAllocated = fir::hasDynamicSize(baseType);
  // Note: temporaries are stack allocated here when possible (do not require
  // stack save/restore) because flang has always stack allocated function
  // results.
  aiir::Value temp = heapAllocated
                         ? builder.createHeapTemporary(loc, baseType, tmpName,
                                                       extents, typeParams)
                         : builder.createTemporary(loc, baseType, tmpName,
                                                   extents, typeParams);
  aiir::Value innerMemory = evalInMem.getMemory();
  temp = builder.createConvert(loc, innerMemory.getType(), temp);
  auto declareOp =
      hlfir::DeclareOp::create(builder, loc, temp, tmpName, shape, typeParams);
  computeEvaluateOpIn(loc, builder, evalInMem, declareOp.getOriginalBase());
  return {hlfir::Entity{declareOp.getBase()}, /*heapAllocated=*/heapAllocated};
}

void hlfir::computeEvaluateOpIn(aiir::Location loc, fir::FirOpBuilder &builder,
                                hlfir::EvaluateInMemoryOp evalInMem,
                                aiir::Value storage) {
  aiir::Value innerMemory = evalInMem.getMemory();
  aiir::Value storageCast =
      builder.createConvert(loc, innerMemory.getType(), storage);
  aiir::IRMapping mapper;
  mapper.map(innerMemory, storageCast);
  for (auto &op : evalInMem.getBody().front().without_terminator())
    builder.clone(op, mapper);
  return;
}

hlfir::Entity hlfir::loadElementAt(aiir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   hlfir::Entity entity,
                                   aiir::ValueRange oneBasedIndices) {
  return loadTrivialScalar(loc, builder,
                           getElementAt(loc, builder, entity, oneBasedIndices));
}

llvm::SmallVector<aiir::Value, Fortran::common::maxRank>
hlfir::genExtentsVector(aiir::Location loc, fir::FirOpBuilder &builder,
                        hlfir::Entity entity) {
  entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
  aiir::Value shape = hlfir::genShape(loc, builder, entity);
  llvm::SmallVector<aiir::Value, Fortran::common::maxRank> extents =
      hlfir::getExplicitExtentsFromShape(shape, builder);
  if (shape.getUses().empty())
    shape.getDefiningOp()->erase();
  return extents;
}

hlfir::Entity hlfir::gen1DSection(aiir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  hlfir::Entity array, int64_t dim,
                                  aiir::ArrayRef<aiir::Value> extents,
                                  aiir::ValueRange oneBasedIndices,
                                  aiir::ArrayRef<aiir::Value> typeParams) {
  assert(array.isVariable() && "array must be a variable");
  assert(dim > 0 && dim <= array.getRank() && "invalid dim number");
  llvm::SmallVector<aiir::Value> lbounds =
      getNonDefaultLowerBounds(loc, builder, array);
  aiir::Value one =
      builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  hlfir::DesignateOp::Subscripts subscripts;
  unsigned indexId = 0;
  for (int i = 0; i < array.getRank(); ++i) {
    if (i == dim - 1) {
      // (...,:, ..)
      if (lbounds.empty()) {
        subscripts.emplace_back(
            hlfir::DesignateOp::Triplet{one, extents[i], one});
      } else {
        aiir::Value ubound =
            genUBound(loc, builder, lbounds[i], extents[i], one);
        subscripts.emplace_back(
            hlfir::DesignateOp::Triplet{lbounds[i], ubound, one});
      }
    } else {
      // (...,lb + one_based_index - 1, ..)
      if (lbounds.empty()) {
        subscripts.emplace_back(oneBasedIndices[indexId++]);
      } else {
        aiir::Value index = genUBound(loc, builder, lbounds[i],
                                      oneBasedIndices[indexId++], one);
        subscripts.emplace_back(index);
      }
    }
  }
  aiir::Value sectionShape =
      fir::ShapeOp::create(builder, loc, extents[dim - 1]);

  // The result type is one of:
  //   !fir.box/class<!fir.array<NxT>>
  //   !fir.box/class<!fir.array<?xT>>
  //
  // We could use !fir.ref<!fir.array<NxT>> when the whole dimension's
  // size is known and it is the leading dimension, but let it be simple
  // for the time being.
  auto seqType =
      aiir::cast<fir::SequenceType>(array.getElementOrSequenceType());
  int64_t dimExtent = seqType.getShape()[dim - 1];
  aiir::Type sectionType =
      fir::SequenceType::get({dimExtent}, seqType.getEleTy());
  sectionType = fir::wrapInClassOrBoxType(sectionType, array.isPolymorphic());

  auto designate = hlfir::DesignateOp::create(
      builder, loc, sectionType, array, /*component=*/"",
      /*componentShape=*/nullptr, subscripts,
      /*substring=*/aiir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
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
        aiir::Value step = subscripts[i++];
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

bool hlfir::isSimplyContiguous(aiir::Value base, bool checkWhole) {
  hlfir::Entity entity{base};
  if (entity.isSimplyContiguous())
    return true;

  // Look at the definition.
  aiir::Operation *def = base.getDefiningOp();
  if (!def)
    return false;

  return aiir::TypeSwitch<aiir::Operation *, bool>(def)
      .Case([&](fir::EmboxOp op) {
        return fir::isContiguousEmbox(op, checkWhole);
      })
      .Case([&](fir::ReboxOp op) {
        hlfir::Entity box{op.getBox()};
        return fir::reboxPreservesContinuity(
                   op, box.mayHaveNonDefaultLowerBounds(), checkWhole) &&
               isSimplyContiguous(box, checkWhole);
      })
      .Case<fir::DeclareOp, hlfir::DeclareOp>([&](auto op) {
        return isSimplyContiguous(op.getMemref(), checkWhole);
      })
      .Case(
          [&](fir::ConvertOp op) { return isSimplyContiguous(op.getValue()); })
      .Default([](auto &&) { return false; });
}
