//===- FIROpenACCUtils.cpp - FIR OpenACC Utilities ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for FIR OpenACC support.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static constexpr llvm::StringRef accPrivateInitName = "acc.private.init";
static constexpr llvm::StringRef accReductionInitName = "acc.reduction.init";

std::string fir::acc::getVariableName(Value v, bool preferDemangledName) {
  std::string srcName;
  std::string prefix;
  llvm::SmallVector<std::string, 4> arrayIndices;
  bool iterate = true;
  mlir::Operation *defOp;

  // For integer constants, no need to further iterate - print their value
  // immediately.
  if (v.getDefiningOp()) {
    IntegerAttr::ValueType val;
    if (matchPattern(v.getDefiningOp(), m_ConstantInt(&val))) {
      llvm::raw_string_ostream os(prefix);
      val.print(os, /*isSigned=*/true);
      return prefix;
    }
  }

  while (v && (defOp = v.getDefiningOp()) && iterate) {
    iterate =
        llvm::TypeSwitch<mlir::Operation *, bool>(defOp)
            .Case<mlir::ViewLikeOpInterface>(
                [&v](mlir::ViewLikeOpInterface op) {
                  v = op.getViewSource();
                  return true;
                })
            .Case<fir::ReboxOp>([&v](fir::ReboxOp op) {
              v = op.getBox();
              return true;
            })
            .Case<fir::EmboxOp>([&v](fir::EmboxOp op) {
              v = op.getMemref();
              return true;
            })
            .Case<fir::ConvertOp>([&v](fir::ConvertOp op) {
              v = op.getValue();
              return true;
            })
            .Case<fir::LoadOp>([&v](fir::LoadOp op) {
              v = op.getMemref();
              return true;
            })
            .Case<fir::BoxAddrOp>([&v](fir::BoxAddrOp op) {
              // The box holds the name of the variable.
              v = op.getVal();
              return true;
            })
            .Case<fir::AddrOfOp>([&](fir::AddrOfOp op) {
              // Only use address_of symbol if mangled name is preferred
              if (!preferDemangledName) {
                auto symRef = op.getSymbol();
                srcName = symRef.getLeafReference().getValue().str();
              }
              return false;
            })
            .Case<fir::ArrayCoorOp>([&](fir::ArrayCoorOp op) {
              v = op.getMemref();
              for (auto coor : op.getIndices()) {
                auto idxName = getVariableName(coor, preferDemangledName);
                arrayIndices.push_back(idxName.empty() ? "?" : idxName);
              }
              return true;
            })
            .Case<fir::CoordinateOp>([&](fir::CoordinateOp op) {
              std::optional<llvm::ArrayRef<int32_t>> fieldIndices =
                  op.getFieldIndices();
              if (fieldIndices && fieldIndices->size() > 0 &&
                  (*fieldIndices)[0] != fir::CoordinateOp::kDynamicIndex) {
                int fieldId = (*fieldIndices)[0];
                mlir::Type baseType =
                    fir::getFortranElementType(op.getRef().getType());
                if (auto recType = llvm::dyn_cast<fir::RecordType>(baseType)) {
                  srcName = recType.getTypeList()[fieldId].first;
                }
              }
              if (!srcName.empty()) {
                // If the field name is known - attempt to continue building
                // name by looking at its parents.
                prefix =
                    getVariableName(op.getRef(), preferDemangledName) + "%";
              }
              return false;
            })
            .Case<hlfir::DesignateOp>([&](hlfir::DesignateOp op) {
              if (op.getComponent()) {
                srcName = op.getComponent().value().str();
                prefix =
                    getVariableName(op.getMemref(), preferDemangledName) + "%";
                return false;
              }
              for (auto coor : op.getIndices()) {
                auto idxName = getVariableName(coor, preferDemangledName);
                arrayIndices.push_back(idxName.empty() ? "?" : idxName);
              }
              v = op.getMemref();
              return true;
            })
            .Case<fir::DeclareOp, hlfir::DeclareOp>([&](auto op) {
              srcName = op.getUniqName().str();
              return false;
            })
            .Case<fir::AllocaOp>([&](fir::AllocaOp op) {
              if (preferDemangledName) {
                // Prefer demangled name (bindc_name over uniq_name)
                srcName = op.getBindcName()  ? *op.getBindcName()
                          : op.getUniqName() ? *op.getUniqName()
                                             : "";
              } else {
                // Prefer mangled name (uniq_name over bindc_name)
                srcName = op.getUniqName()    ? *op.getUniqName()
                          : op.getBindcName() ? *op.getBindcName()
                                              : "";
              }
              return false;
            })
            .Default([](mlir::Operation *) { return false; });
  }

  // Fallback to the default implementation.
  if (srcName.empty())
    return mlir::acc::getVariableName(v);

  // Build array index suffix if present
  std::string suffix;
  if (!arrayIndices.empty()) {
    llvm::raw_string_ostream os(suffix);
    os << "(";
    llvm::interleaveComma(arrayIndices, os);
    os << ")";
  }

  // Names from FIR operations may be mangled.
  // When the demangled name is requested - demangle it.
  if (preferDemangledName) {
    auto [kind, deconstructed] = fir::NameUniquer::deconstruct(srcName);
    if (kind != fir::NameUniquer::NameKind::NOT_UNIQUED)
      return prefix + deconstructed.name + suffix;
  }

  return prefix + srcName + suffix;
}

bool fir::acc::areAllBoundsConstant(llvm::ArrayRef<Value> bounds) {
  for (auto bound : bounds) {
    auto dataBound =
        mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
    if (!dataBound)
      return false;

    // Check if this bound has constant values
    bool hasConstant = false;
    if (dataBound.getLowerbound() && dataBound.getUpperbound())
      hasConstant =
          fir::getIntIfConstant(dataBound.getLowerbound()).has_value() &&
          fir::getIntIfConstant(dataBound.getUpperbound()).has_value();
    else if (dataBound.getExtent())
      hasConstant = fir::getIntIfConstant(dataBound.getExtent()).has_value();

    if (!hasConstant)
      return false;
  }
  return true;
}

static std::string getBoundsString(llvm::ArrayRef<Value> bounds) {
  if (bounds.empty())
    return "";

  std::string boundStr;
  llvm::raw_string_ostream os(boundStr);
  os << "_section_";

  llvm::interleave(
      bounds,
      [&](Value bound) {
        auto boundsOp =
            mlir::cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
        if (boundsOp.getLowerbound() &&
            fir::getIntIfConstant(boundsOp.getLowerbound()) &&
            boundsOp.getUpperbound() &&
            fir::getIntIfConstant(boundsOp.getUpperbound())) {
          os << "lb" << *fir::getIntIfConstant(boundsOp.getLowerbound())
             << ".ub" << *fir::getIntIfConstant(boundsOp.getUpperbound());
        } else if (boundsOp.getExtent() &&
                   fir::getIntIfConstant(boundsOp.getExtent())) {
          os << "ext" << *fir::getIntIfConstant(boundsOp.getExtent());
        } else {
          os << "?";
        }
      },
      [&] { os << "x"; });

  return os.str();
}

static std::string getRecipeName(mlir::acc::RecipeKind kind, Type type,
                                 const fir::KindMapping &kindMap,
                                 llvm::ArrayRef<Value> bounds,
                                 mlir::acc::ReductionOperator reductionOp =
                                     mlir::acc::ReductionOperator::AccNone) {
  assert(fir::isa_fir_type(type) && "getRecipeName expects a FIR type");

  // Build the complete prefix with all components before calling
  // getTypeAsString
  std::string prefixStr;
  llvm::raw_string_ostream prefixOS(prefixStr);

  switch (kind) {
  case mlir::acc::RecipeKind::private_recipe:
    prefixOS << "privatization";
    break;
  case mlir::acc::RecipeKind::firstprivate_recipe:
    prefixOS << "firstprivatization";
    break;
  case mlir::acc::RecipeKind::reduction_recipe:
    prefixOS << "reduction";
    // Embed the reduction operator in the prefix
    if (reductionOp != mlir::acc::ReductionOperator::AccNone)
      prefixOS << "_"
               << mlir::acc::stringifyReductionOperator(reductionOp).str();
    break;
  }

  if (!bounds.empty())
    prefixOS << getBoundsString(bounds);

  return fir::getTypeAsString(type, kindMap, prefixOS.str());
}

std::string fir::acc::getRecipeName(mlir::acc::RecipeKind kind, Type type,
                                    Value var, llvm::ArrayRef<Value> bounds,
                                    mlir::acc::ReductionOperator reductionOp) {
  auto kindMap = var && var.getDefiningOp()
                     ? fir::getKindMapping(var.getDefiningOp())
                     : fir::KindMapping(type.getContext());
  return ::getRecipeName(kind, type, kindMap, bounds, reductionOp);
}

/// Get the initial value for reduction operator.
template <typename R>
static R getReductionInitValue(mlir::acc::ReductionOperator op, mlir::Type ty) {
  if (op == mlir::acc::ReductionOperator::AccMin) {
    // min init value -> largest
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt::getSignedMaxValue(ty.getIntOrFloatBitWidth());
    }
    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty);
      assert(floatTy && "expect float type");
      return llvm::APFloat::getLargest(floatTy.getFloatSemantics(),
                                       /*negative=*/false);
    }
  } else if (op == mlir::acc::ReductionOperator::AccMax) {
    // max init value -> smallest
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt::getSignedMinValue(ty.getIntOrFloatBitWidth());
    }
    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty);
      assert(floatTy && "expect float type");
      return llvm::APFloat::getSmallest(floatTy.getFloatSemantics(),
                                        /*negative=*/true);
    }
  } else if (op == mlir::acc::ReductionOperator::AccIand) {
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer type");
      unsigned bits = ty.getIntOrFloatBitWidth();
      return llvm::APInt::getAllOnes(bits);
    }
  } else {
    assert(op != mlir::acc::ReductionOperator::AccNone);
    // +, ior, ieor init value -> 0
    // * init value -> 1
    int64_t value = (op == mlir::acc::ReductionOperator::AccMul) ? 1 : 0;
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt(ty.getIntOrFloatBitWidth(), value, true);
    }

    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      assert(mlir::isa<mlir::FloatType>(ty) && "expect float type");
      auto floatTy = mlir::dyn_cast<mlir::FloatType>(ty);
      return llvm::APFloat(floatTy.getFloatSemantics(), value);
    }

    if constexpr (std::is_same_v<R, int64_t>)
      return value;
  }
  llvm_unreachable("OpenACC reduction unsupported type");
}

/// Return a constant with the initial value for the reduction operator and
/// type combination.
static mlir::Value getReductionInitValue(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Type varType,
                                         mlir::acc::ReductionOperator op) {
  mlir::Type ty = fir::getFortranElementType(varType);
  if (op == mlir::acc::ReductionOperator::AccLand ||
      op == mlir::acc::ReductionOperator::AccLor ||
      op == mlir::acc::ReductionOperator::AccEqv ||
      op == mlir::acc::ReductionOperator::AccNeqv) {
    assert(mlir::isa<fir::LogicalType>(ty) && "expect fir.logical type");
    bool value = true; // .true. for .and. and .eqv.
    if (op == mlir::acc::ReductionOperator::AccLor ||
        op == mlir::acc::ReductionOperator::AccNeqv)
      value = false; // .false. for .or. and .neqv.
    return builder.createBool(loc, value);
  }
  if (ty.isIntOrIndex())
    return mlir::arith::ConstantOp::create(
        builder, loc, ty,
        builder.getIntegerAttr(ty, getReductionInitValue<llvm::APInt>(op, ty)));
  if (op == mlir::acc::ReductionOperator::AccMin ||
      op == mlir::acc::ReductionOperator::AccMax) {
    if (mlir::isa<mlir::ComplexType>(ty))
      llvm::report_fatal_error(
          "min/max reduction not supported for complex type");
    if (auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty))
      return mlir::arith::ConstantOp::create(
          builder, loc, ty,
          builder.getFloatAttr(ty,
                               getReductionInitValue<llvm::APFloat>(op, ty)));
  } else if (auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty)) {
    return mlir::arith::ConstantOp::create(
        builder, loc, ty,
        builder.getFloatAttr(ty, getReductionInitValue<int64_t>(op, ty)));
  } else if (auto cmplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(ty)) {
    mlir::Type floatTy = cmplxTy.getElementType();
    mlir::Value realInit = builder.createRealConstant(
        loc, floatTy, getReductionInitValue<int64_t>(op, cmplxTy));
    mlir::Value imagInit = builder.createRealConstant(loc, floatTy, 0.0);
    return fir::factory::Complex{builder, loc}.createComplex(cmplxTy, realInit,
                                                             imagInit);
  }
  llvm::report_fatal_error("Unsupported OpenACC reduction type");
}

static llvm::SmallVector<mlir::Value>
getRecipeBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::ValueRange dataBoundOps,
                mlir::ValueRange blockBoundArgs) {
  if (dataBoundOps.empty())
    return {};
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  llvm::SmallVector<mlir::Value> bounds;
  if (!blockBoundArgs.empty()) {
    for (unsigned i = 0; i + 2 < blockBoundArgs.size(); i += 3) {
      bounds.push_back(blockBoundArgs[i]);
      bounds.push_back(blockBoundArgs[i + 1]);
      // acc data bound strides is the inner size in bytes or elements, but
      // sections are always 1-based, so there is no need to try to compute
      // that back from the acc bounds.
      bounds.push_back(one);
    }
    return bounds;
  }
  for (auto bound : dataBoundOps) {
    auto dataBound = llvm::dyn_cast_if_present<mlir::acc::DataBoundsOp>(
        bound.getDefiningOp());
    assert(dataBound && "expect acc bounds to be produced by DataBoundsOp");
    assert(
        dataBound.getLowerbound() && dataBound.getUpperbound() &&
        "expect acc bounds for Fortran to always have lower and upper bounds");
    std::optional<std::int64_t> lb =
        fir::getIntIfConstant(dataBound.getLowerbound());
    std::optional<std::int64_t> ub =
        fir::getIntIfConstant(dataBound.getUpperbound());
    assert(lb.has_value() && ub.has_value() &&
           "must get constant bounds when there are no bound block arguments");
    bounds.push_back(builder.createIntegerConstant(loc, idxTy, *lb));
    bounds.push_back(builder.createIntegerConstant(loc, idxTy, *ub));
    bounds.push_back(one);
  }
  return bounds;
}

static void addRecipeBoundsArgs(llvm::SmallVector<mlir::Value> &bounds,
                                bool allConstantBound,
                                llvm::SmallVector<mlir::Type> &argsTy,
                                llvm::SmallVector<mlir::Location> &argsLoc) {
  if (!allConstantBound) {
    for (mlir::Value bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      argsTy.push_back(dataBound.getLowerbound().getType());
      argsLoc.push_back(dataBound.getLowerbound().getLoc());
      argsTy.push_back(dataBound.getUpperbound().getType());
      argsLoc.push_back(dataBound.getUpperbound().getLoc());
      argsTy.push_back(dataBound.getStartIdx().getType());
      argsLoc.push_back(dataBound.getStartIdx().getLoc());
    }
  }
}

using MappableValue = mlir::TypedValue<mlir::acc::MappableType>;

// Generate the combiner or copy region block and block arguments and return the
// source and destination entities.
static std::pair<MappableValue, MappableValue>
genRecipeCombinerOrCopyRegion(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Type ty, mlir::Region &region,
                              llvm::SmallVector<mlir::Value> &bounds,
                              bool allConstantBound) {
  llvm::SmallVector<mlir::Type> argsTy{ty, ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc, loc};
  addRecipeBoundsArgs(bounds, allConstantBound, argsTy, argsLoc);
  mlir::Block *block =
      builder.createBlock(&region, region.end(), argsTy, argsLoc);
  builder.setInsertionPointToEnd(&region.back());
  auto firstArg = mlir::cast<MappableValue>(block->getArgument(0));
  auto secondArg = mlir::cast<MappableValue>(block->getArgument(1));
  return {firstArg, secondArg};
}

template <typename RecipeOp>
static RecipeOp genRecipeOp(
    fir::FirOpBuilder &builder, mlir::ModuleOp mod, llvm::StringRef recipeName,
    mlir::Location loc, mlir::Type ty,
    llvm::SmallVector<mlir::Value> &dataOperationBounds, bool allConstantBound,
    mlir::acc::ReductionOperator op = mlir::acc::ReductionOperator::AccNone) {
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  RecipeOp recipe;
  if constexpr (std::is_same_v<RecipeOp, mlir::acc::ReductionRecipeOp>) {
    recipe = mlir::acc::ReductionRecipeOp::create(modBuilder, loc, recipeName,
                                                  ty, op);
  } else {
    recipe = RecipeOp::create(modBuilder, loc, recipeName, ty);
  }

  assert(hlfir::isFortranVariableType(ty) && "expect Fortran variable type");

  llvm::SmallVector<mlir::Type> argsTy{ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc};
  if (!dataOperationBounds.empty())
    addRecipeBoundsArgs(dataOperationBounds, allConstantBound, argsTy, argsLoc);

  auto initBlock = builder.createBlock(
      &recipe.getInitRegion(), recipe.getInitRegion().end(), argsTy, argsLoc);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  mlir::Value initValue;
  if constexpr (std::is_same_v<RecipeOp, mlir::acc::ReductionRecipeOp>) {
    assert(op != mlir::acc::ReductionOperator::AccNone);
    initValue = getReductionInitValue(builder, loc, ty, op);
  }

  // Since we reuse the same recipe for all variables of the same type - we
  // cannot use the actual variable name. Thus use a temporary name.
  llvm::StringRef initName;
  if constexpr (std::is_same_v<RecipeOp, mlir::acc::ReductionRecipeOp>)
    initName = accReductionInitName;
  else
    initName = accPrivateInitName;

  auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  bool needsDestroy = false;
  llvm::SmallVector<mlir::Value> initBounds =
      getRecipeBounds(builder, loc, dataOperationBounds,
                      initBlock->getArguments().drop_front(1));
  mlir::Value retVal = mappableTy.generatePrivateInit(
      builder, loc, mlir::cast<MappableValue>(initBlock->getArgument(0)),
      initName, initBounds, initValue, needsDestroy);
  mlir::acc::YieldOp::create(builder, loc, retVal);
  // Create destroy region and generate destruction if requested.
  if (needsDestroy) {
    llvm::SmallVector<mlir::Type> destroyArgsTy;
    llvm::SmallVector<mlir::Location> destroyArgsLoc;
    // original and privatized/reduction value
    destroyArgsTy.push_back(ty);
    destroyArgsTy.push_back(ty);
    destroyArgsLoc.push_back(loc);
    destroyArgsLoc.push_back(loc);
    // Append bounds arguments (if any) in the same order as init region
    if (argsTy.size() > 1) {
      destroyArgsTy.append(argsTy.begin() + 1, argsTy.end());
      destroyArgsLoc.insert(destroyArgsLoc.end(), argsTy.size() - 1, loc);
    }

    mlir::Block *destroyBlock = builder.createBlock(
        &recipe.getDestroyRegion(), recipe.getDestroyRegion().end(),
        destroyArgsTy, destroyArgsLoc);
    builder.setInsertionPointToEnd(destroyBlock);

    llvm::SmallVector<mlir::Value> destroyBounds =
        getRecipeBounds(builder, loc, dataOperationBounds,
                        destroyBlock->getArguments().drop_front(2));
    [[maybe_unused]] bool success = mappableTy.generatePrivateDestroy(
        builder, loc, destroyBlock->getArgument(1), destroyBounds);
    assert(success && "failed to generate destroy region");
    mlir::acc::TerminatorOp::create(builder, loc);
  }
  return recipe;
}

mlir::SymbolRefAttr
fir::acc::createOrGetPrivateRecipe(mlir::OpBuilder &mlirBuilder,
                                   mlir::Location loc, mlir::Type ty,
                                   llvm::SmallVector<mlir::Value> &bounds) {
  mlir::ModuleOp mod =
      mlirBuilder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(mlirBuilder, mod);
  std::string recipeName = ::getRecipeName(
      mlir::acc::RecipeKind::private_recipe, ty, builder.getKindMap(), bounds);
  if (auto recipe = mod.lookupSymbol<mlir::acc::PrivateRecipeOp>(recipeName))
    return mlir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());

  mlir::OpBuilder::InsertionGuard guard(builder);
  bool allConstantBound = fir::acc::areAllBoundsConstant(bounds);
  auto recipe = genRecipeOp<mlir::acc::PrivateRecipeOp>(
      builder, mod, recipeName, loc, ty, bounds, allConstantBound);
  return mlir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());
}

mlir::SymbolRefAttr fir::acc::createOrGetFirstprivateRecipe(
    mlir::OpBuilder &mlirBuilder, mlir::Location loc, mlir::Type ty,
    llvm::SmallVector<mlir::Value> &dataBoundOps) {
  mlir::ModuleOp mod =
      mlirBuilder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(mlirBuilder, mod);
  std::string recipeName =
      ::getRecipeName(mlir::acc::RecipeKind::firstprivate_recipe, ty,
                      builder.getKindMap(), dataBoundOps);
  if (auto recipe =
          mod.lookupSymbol<mlir::acc::FirstprivateRecipeOp>(recipeName))
    return mlir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());

  mlir::OpBuilder::InsertionGuard guard(builder);
  bool allConstantBound = fir::acc::areAllBoundsConstant(dataBoundOps);
  auto recipe = genRecipeOp<mlir::acc::FirstprivateRecipeOp>(
      builder, mod, recipeName, loc, ty, dataBoundOps, allConstantBound);
  auto [source, destination] = genRecipeCombinerOrCopyRegion(
      builder, loc, ty, recipe.getCopyRegion(), dataBoundOps, allConstantBound);
  llvm::SmallVector<mlir::Value> copyBounds =
      getRecipeBounds(builder, loc, dataBoundOps,
                      recipe.getCopyRegion().getArguments().drop_front(2));

  auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  [[maybe_unused]] bool success =
      mappableTy.generateCopy(builder, loc, source, destination, copyBounds);
  assert(success && "failed to generate copy");
  mlir::acc::TerminatorOp::create(builder, loc);
  return mlir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());
}

mlir::SymbolRefAttr fir::acc::createOrGetReductionRecipe(
    mlir::OpBuilder &mlirBuilder, mlir::Location loc, mlir::Type ty,
    mlir::acc::ReductionOperator op,
    llvm::SmallVector<mlir::Value> &dataBoundOps,
    mlir::Attribute fastMathAttr) {
  mlir::ModuleOp mod =
      mlirBuilder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(mlirBuilder, mod);
  std::string recipeName =
      ::getRecipeName(mlir::acc::RecipeKind::reduction_recipe, ty,
                      builder.getKindMap(), dataBoundOps, op);
  if (auto recipe = mod.lookupSymbol<mlir::acc::ReductionRecipeOp>(recipeName))
    return mlir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());

  mlir::OpBuilder::InsertionGuard guard(builder);
  bool allConstantBound = fir::acc::areAllBoundsConstant(dataBoundOps);
  auto recipe = genRecipeOp<mlir::acc::ReductionRecipeOp>(
      builder, mod, recipeName, loc, ty, dataBoundOps, allConstantBound, op);

  auto [dest, source] = genRecipeCombinerOrCopyRegion(
      builder, loc, ty, recipe.getCombinerRegion(), dataBoundOps,
      allConstantBound);
  llvm::SmallVector<mlir::Value> combinerBounds =
      getRecipeBounds(builder, loc, dataBoundOps,
                      recipe.getCombinerRegion().getArguments().drop_front(2));

  auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  [[maybe_unused]] bool success = mappableTy.generateCombiner(
      builder, loc, dest, source, combinerBounds, op, fastMathAttr);
  assert(success && "failed to generate combiner");
  mlir::acc::YieldOp::create(builder, loc, dest);
  return mlir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());
}
