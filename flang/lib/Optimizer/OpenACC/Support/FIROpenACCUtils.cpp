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
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/OpenACC/OpenACCUtils.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace aiir;

static constexpr llvm::StringRef accPrivateInitName = "acc.private.init";
static constexpr llvm::StringRef accReductionInitName = "acc.reduction.init";

std::string fir::acc::getVariableName(Value v, bool preferDemangledName) {
  std::string srcName;
  std::string prefix;
  llvm::SmallVector<std::string, 4> arrayIndices;
  bool iterate = true;
  aiir::Operation *defOp;

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
        llvm::TypeSwitch<aiir::Operation *, bool>(defOp)
            .Case([&v](aiir::ViewLikeOpInterface op) {
              v = op.getViewSource();
              return true;
            })
            .Case([&v](fir::ReboxOp op) {
              v = op.getBox();
              return true;
            })
            .Case([&v](fir::EmboxOp op) {
              v = op.getMemref();
              return true;
            })
            .Case([&v](fir::ConvertOp op) {
              v = op.getValue();
              return true;
            })
            .Case([&v](fir::LoadOp op) {
              v = op.getMemref();
              return true;
            })
            .Case([&v](fir::BoxAddrOp op) {
              // The box holds the name of the variable.
              v = op.getVal();
              return true;
            })
            .Case([&](fir::AddrOfOp op) {
              // Only use address_of symbol if mangled name is preferred
              if (!preferDemangledName) {
                auto symRef = op.getSymbol();
                srcName = symRef.getLeafReference().getValue().str();
              }
              return false;
            })
            .Case([&](fir::ArrayCoorOp op) {
              v = op.getMemref();
              for (auto coor : op.getIndices()) {
                auto idxName = getVariableName(coor, preferDemangledName);
                arrayIndices.push_back(idxName.empty() ? "?" : idxName);
              }
              return true;
            })
            .Case([&](fir::CoordinateOp op) {
              std::optional<llvm::ArrayRef<int32_t>> fieldIndices =
                  op.getFieldIndices();
              if (fieldIndices && fieldIndices->size() > 0 &&
                  (*fieldIndices)[0] != fir::CoordinateOp::kDynamicIndex) {
                int fieldId = (*fieldIndices)[0];
                aiir::Type baseType =
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
            .Case([&](hlfir::DesignateOp op) {
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
            .Case([&](fir::AllocaOp op) {
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
            .Default([](aiir::Operation *) { return false; });
  }

  // Fallback to the default implementation.
  if (srcName.empty())
    return aiir::acc::getVariableName(v);

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
        aiir::dyn_cast<aiir::acc::DataBoundsOp>(bound.getDefiningOp());
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
            aiir::cast<aiir::acc::DataBoundsOp>(bound.getDefiningOp());
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

static std::string getRecipeName(aiir::acc::RecipeKind kind, Type type,
                                 aiir::acc::VariableInfoAttr varInfo,
                                 const fir::KindMapping &kindMap,
                                 llvm::ArrayRef<Value> bounds,
                                 aiir::acc::ReductionOperator reductionOp =
                                     aiir::acc::ReductionOperator::AccNone) {
  assert(fir::isa_fir_type(type) && "getRecipeName expects a FIR type");

  // Build the complete prefix with all components before calling
  // getTypeAsString
  std::string prefixStr;
  llvm::raw_string_ostream prefixOS(prefixStr);

  switch (kind) {
  case aiir::acc::RecipeKind::private_recipe:
    prefixOS << "privatization";
    break;
  case aiir::acc::RecipeKind::firstprivate_recipe:
    prefixOS << "firstprivatization";
    break;
  case aiir::acc::RecipeKind::reduction_recipe:
    prefixOS << "reduction";
    // Embed the reduction operator in the prefix
    if (reductionOp != aiir::acc::ReductionOperator::AccNone)
      prefixOS << "_"
               << aiir::acc::stringifyReductionOperator(reductionOp).str();
    break;
  }

  if (auto fortranVarInfo =
          aiir::dyn_cast_if_present<fir::OpenACCFortranVariableInfoAttr>(
              varInfo))
    if (fortranVarInfo.getMayBeOptional())
      prefixOS << "_optional";

  if (!bounds.empty())
    prefixOS << getBoundsString(bounds);

  return fir::getTypeAsString(type, kindMap, prefixOS.str());
}

using MappableValue = aiir::TypedValue<aiir::acc::MappableType>;

std::string fir::acc::getRecipeName(aiir::acc::RecipeKind kind, Type type,
                                    Value var, llvm::ArrayRef<Value> bounds,
                                    aiir::acc::ReductionOperator reductionOp) {
  auto kindMap = var && var.getDefiningOp()
                     ? fir::getKindMapping(var.getDefiningOp())
                     : fir::KindMapping(type.getContext());
  aiir::acc::VariableInfoAttr varInfo;
  if (var)
    if (auto mappableTy =
            aiir::dyn_cast<aiir::acc::MappableType>(var.getType()))
      varInfo =
          mappableTy.genPrivateVariableInfo(aiir::cast<MappableValue>(var));
  return ::getRecipeName(kind, type, varInfo, kindMap, bounds, reductionOp);
}

/// Map acc::ReductionOperator to arith::AtomicRMWKind for identity value
/// computation. Uses minimumf/maximumf instead of minnumf/maxnumf because
/// arith::getIdentityValueAttr for minnumf/maxnumf returns NaN (the IEEE 754
/// identity), which doesn't work with comparison-based reductions on GPU.
/// minimumf/maximumf identity with useOnlyFiniteValue gives the correct
/// finite extreme value (FLT_MAX / -FLT_MAX).
static aiir::arith::AtomicRMWKind
getAtomicRMWKindForIdentity(aiir::acc::ReductionOperator op, aiir::Type ty) {
  bool isFloat = aiir::isa<aiir::FloatType>(ty);
  switch (op) {
  case aiir::acc::ReductionOperator::AccAdd:
    return isFloat ? aiir::arith::AtomicRMWKind::addf
                   : aiir::arith::AtomicRMWKind::addi;
  case aiir::acc::ReductionOperator::AccMul:
    return isFloat ? aiir::arith::AtomicRMWKind::mulf
                   : aiir::arith::AtomicRMWKind::muli;
  case aiir::acc::ReductionOperator::AccMin:
  case aiir::acc::ReductionOperator::AccMinnumf:
  case aiir::acc::ReductionOperator::AccMinimumf:
    return isFloat ? aiir::arith::AtomicRMWKind::minimumf
                   : aiir::arith::AtomicRMWKind::mins;
  case aiir::acc::ReductionOperator::AccMax:
  case aiir::acc::ReductionOperator::AccMaxnumf:
  case aiir::acc::ReductionOperator::AccMaximumf:
    return isFloat ? aiir::arith::AtomicRMWKind::maximumf
                   : aiir::arith::AtomicRMWKind::maxs;
  case aiir::acc::ReductionOperator::AccIand:
    return aiir::arith::AtomicRMWKind::andi;
  case aiir::acc::ReductionOperator::AccIor:
    return aiir::arith::AtomicRMWKind::ori;
  case aiir::acc::ReductionOperator::AccXor:
    return aiir::arith::AtomicRMWKind::xori;
  default:
    llvm_unreachable("unsupported acc::ReductionOperator");
  }
}

/// Return a constant with the initial value for the reduction operator and
/// type combination.
static aiir::Value getReductionInitValue(fir::FirOpBuilder &builder,
                                         aiir::Location loc, aiir::Type varType,
                                         aiir::acc::ReductionOperator op) {
  aiir::Type ty = fir::getFortranElementType(varType);
  if (op == aiir::acc::ReductionOperator::AccLand ||
      op == aiir::acc::ReductionOperator::AccLor ||
      op == aiir::acc::ReductionOperator::AccEqv ||
      op == aiir::acc::ReductionOperator::AccNeqv) {
    assert(aiir::isa<fir::LogicalType>(ty) && "expect fir.logical type");
    bool value = (op == aiir::acc::ReductionOperator::AccLand ||
                  op == aiir::acc::ReductionOperator::AccEqv);
    return builder.createBool(loc, value);
  }
  if (auto cmplxTy = aiir::dyn_cast<aiir::ComplexType>(ty)) {
    aiir::arith::AtomicRMWKind kind =
        getAtomicRMWKindForIdentity(op, cmplxTy.getElementType());
    aiir::Value realInit = aiir::arith::getIdentityValue(
        kind, cmplxTy.getElementType(), builder, loc,
        /*useOnlyFiniteValue=*/true);
    aiir::Value imagInit =
        builder.createRealConstant(loc, cmplxTy.getElementType(), 0.0);
    return fir::factory::Complex{builder, loc}.createComplex(cmplxTy, realInit,
                                                             imagInit);
  }
  aiir::arith::AtomicRMWKind kind = getAtomicRMWKindForIdentity(op, ty);
  return aiir::arith::getIdentityValue(kind, ty, builder, loc,
                                       /*useOnlyFiniteValue=*/true);
}

static llvm::SmallVector<aiir::Value>
getRecipeBounds(fir::FirOpBuilder &builder, aiir::Location loc,
                aiir::ValueRange dataBoundOps,
                aiir::ValueRange blockBoundArgs) {
  if (dataBoundOps.empty())
    return {};
  aiir::Type idxTy = builder.getIndexType();
  aiir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  llvm::SmallVector<aiir::Value> bounds;
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
    auto dataBound = llvm::dyn_cast_if_present<aiir::acc::DataBoundsOp>(
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

static void addRecipeBoundsArgs(llvm::SmallVector<aiir::Value> &bounds,
                                bool allConstantBound,
                                llvm::SmallVector<aiir::Type> &argsTy,
                                llvm::SmallVector<aiir::Location> &argsLoc) {
  if (!allConstantBound) {
    for (aiir::Value bound : llvm::reverse(bounds)) {
      auto dataBound =
          aiir::dyn_cast<aiir::acc::DataBoundsOp>(bound.getDefiningOp());
      argsTy.push_back(dataBound.getLowerbound().getType());
      argsLoc.push_back(dataBound.getLowerbound().getLoc());
      argsTy.push_back(dataBound.getUpperbound().getType());
      argsLoc.push_back(dataBound.getUpperbound().getLoc());
      argsTy.push_back(dataBound.getStartIdx().getType());
      argsLoc.push_back(dataBound.getStartIdx().getLoc());
    }
  }
}

// Generate the combiner or copy region block and block arguments and return the
// source and destination entities.
static std::pair<MappableValue, MappableValue>
genRecipeCombinerOrCopyRegion(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::Type ty, aiir::Region &region,
                              llvm::SmallVector<aiir::Value> &bounds,
                              bool allConstantBound) {
  llvm::SmallVector<aiir::Type> argsTy{ty, ty};
  llvm::SmallVector<aiir::Location> argsLoc{loc, loc};
  addRecipeBoundsArgs(bounds, allConstantBound, argsTy, argsLoc);
  aiir::Block *block =
      builder.createBlock(&region, region.end(), argsTy, argsLoc);
  builder.setInsertionPointToEnd(&region.back());
  auto firstArg = aiir::cast<MappableValue>(block->getArgument(0));
  auto secondArg = aiir::cast<MappableValue>(block->getArgument(1));
  return {firstArg, secondArg};
}

template <typename RecipeOp>
static RecipeOp genRecipeOp(
    fir::FirOpBuilder &builder, aiir::ModuleOp mod, llvm::StringRef recipeName,
    aiir::Location loc, aiir::Type ty, aiir::acc::VariableInfoAttr varInfo,
    llvm::SmallVector<aiir::Value> &dataOperationBounds, bool allConstantBound,
    aiir::acc::ReductionOperator op = aiir::acc::ReductionOperator::AccNone) {
  aiir::OpBuilder modBuilder(mod.getBodyRegion());
  RecipeOp recipe;
  if constexpr (std::is_same_v<RecipeOp, aiir::acc::ReductionRecipeOp>) {
    recipe = aiir::acc::ReductionRecipeOp::create(modBuilder, loc, recipeName,
                                                  ty, op);
  } else {
    recipe = RecipeOp::create(modBuilder, loc, recipeName, ty);
  }

  assert(hlfir::isFortranVariableType(ty) && "expect Fortran variable type");

  llvm::SmallVector<aiir::Type> argsTy{ty};
  llvm::SmallVector<aiir::Location> argsLoc{loc};
  if (!dataOperationBounds.empty())
    addRecipeBoundsArgs(dataOperationBounds, allConstantBound, argsTy, argsLoc);

  auto initBlock = builder.createBlock(
      &recipe.getInitRegion(), recipe.getInitRegion().end(), argsTy, argsLoc);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  aiir::Value initValue;
  if constexpr (std::is_same_v<RecipeOp, aiir::acc::ReductionRecipeOp>) {
    assert(op != aiir::acc::ReductionOperator::AccNone);
    initValue = getReductionInitValue(builder, loc, ty, op);
  }

  // Since we reuse the same recipe for all variables of the same type - we
  // cannot use the actual variable name. Thus use a temporary name.
  llvm::StringRef initName;
  if constexpr (std::is_same_v<RecipeOp, aiir::acc::ReductionRecipeOp>)
    initName = accReductionInitName;
  else
    initName = accPrivateInitName;

  auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  auto initArg = aiir::cast<MappableValue>(initBlock->getArgument(0));
  bool needsDestroy = false;
  llvm::SmallVector<aiir::Value> initBounds =
      getRecipeBounds(builder, loc, dataOperationBounds,
                      initBlock->getArguments().drop_front(1));
  aiir::Value retVal = mappableTy.generatePrivateInit(
      builder, loc, initArg, initName, initBounds, initValue, varInfo,
      needsDestroy);
  aiir::acc::YieldOp::create(builder, loc, retVal);
  // Create destroy region and generate destruction if requested.
  if (needsDestroy) {
    llvm::SmallVector<aiir::Type> destroyArgsTy;
    llvm::SmallVector<aiir::Location> destroyArgsLoc;
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

    aiir::Block *destroyBlock = builder.createBlock(
        &recipe.getDestroyRegion(), recipe.getDestroyRegion().end(),
        destroyArgsTy, destroyArgsLoc);
    builder.setInsertionPointToEnd(destroyBlock);

    llvm::SmallVector<aiir::Value> destroyBounds =
        getRecipeBounds(builder, loc, dataOperationBounds,
                        destroyBlock->getArguments().drop_front(2));
    [[maybe_unused]] bool success = mappableTy.generatePrivateDestroy(
        builder, loc, destroyBlock->getArgument(1), destroyBounds, varInfo);
    assert(success && "failed to generate destroy region");
    aiir::acc::TerminatorOp::create(builder, loc);
  }
  return recipe;
}

aiir::SymbolRefAttr
fir::acc::createOrGetPrivateRecipe(aiir::OpBuilder &aiirBuilder,
                                   aiir::Location loc, aiir::Value var,
                                   llvm::SmallVector<aiir::Value> &bounds) {
  aiir::Type ty = var.getType();
  aiir::ModuleOp mod =
      aiirBuilder.getBlock()->getParent()->getParentOfType<aiir::ModuleOp>();
  fir::FirOpBuilder builder(aiirBuilder, mod);
  auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  aiir::acc::VariableInfoAttr varInfo =
      mappableTy.genPrivateVariableInfo(aiir::cast<MappableValue>(var));
  std::string recipeName =
      ::getRecipeName(aiir::acc::RecipeKind::private_recipe, ty, varInfo,
                      builder.getKindMap(), bounds);
  if (auto recipe = mod.lookupSymbol<aiir::acc::PrivateRecipeOp>(recipeName))
    return aiir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());

  aiir::OpBuilder::InsertionGuard guard(builder);
  bool allConstantBound = fir::acc::areAllBoundsConstant(bounds);
  auto recipe = genRecipeOp<aiir::acc::PrivateRecipeOp>(
      builder, mod, recipeName, loc, ty, varInfo, bounds, allConstantBound);
  return aiir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());
}

aiir::SymbolRefAttr fir::acc::createOrGetFirstprivateRecipe(
    aiir::OpBuilder &aiirBuilder, aiir::Location loc, aiir::Value var,
    llvm::SmallVector<aiir::Value> &dataBoundOps) {
  aiir::Type ty = var.getType();
  aiir::ModuleOp mod =
      aiirBuilder.getBlock()->getParent()->getParentOfType<aiir::ModuleOp>();
  fir::FirOpBuilder builder(aiirBuilder, mod);
  auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  aiir::acc::VariableInfoAttr varInfo =
      mappableTy.genPrivateVariableInfo(aiir::cast<MappableValue>(var));
  std::string recipeName =
      ::getRecipeName(aiir::acc::RecipeKind::firstprivate_recipe, ty, varInfo,
                      builder.getKindMap(), dataBoundOps);
  if (auto recipe =
          mod.lookupSymbol<aiir::acc::FirstprivateRecipeOp>(recipeName))
    return aiir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());

  aiir::OpBuilder::InsertionGuard guard(builder);
  bool allConstantBound = fir::acc::areAllBoundsConstant(dataBoundOps);
  auto recipe = genRecipeOp<aiir::acc::FirstprivateRecipeOp>(
      builder, mod, recipeName, loc, ty, varInfo, dataBoundOps,
      allConstantBound);
  auto [source, destination] = genRecipeCombinerOrCopyRegion(
      builder, loc, ty, recipe.getCopyRegion(), dataBoundOps, allConstantBound);
  llvm::SmallVector<aiir::Value> copyBounds =
      getRecipeBounds(builder, loc, dataBoundOps,
                      recipe.getCopyRegion().getArguments().drop_front(2));

  [[maybe_unused]] bool success = mappableTy.generateCopy(
      builder, loc, source, destination, copyBounds, varInfo);
  assert(success && "failed to generate copy");
  aiir::acc::TerminatorOp::create(builder, loc);
  return aiir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());
}

aiir::SymbolRefAttr fir::acc::createOrGetReductionRecipe(
    aiir::OpBuilder &aiirBuilder, aiir::Location loc, aiir::Value var,
    aiir::acc::ReductionOperator op,
    llvm::SmallVector<aiir::Value> &dataBoundOps,
    aiir::Attribute fastMathAttr) {
  aiir::Type ty = var.getType();
  aiir::ModuleOp mod =
      aiirBuilder.getBlock()->getParent()->getParentOfType<aiir::ModuleOp>();
  fir::FirOpBuilder builder(aiirBuilder, mod);
  auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(ty);
  assert(mappableTy &&
         "Expected that all variable types are considered mappable");
  aiir::acc::VariableInfoAttr varInfo =
      mappableTy.genPrivateVariableInfo(aiir::cast<MappableValue>(var));
  std::string recipeName =
      ::getRecipeName(aiir::acc::RecipeKind::reduction_recipe, ty, varInfo,
                      builder.getKindMap(), dataBoundOps, op);
  if (auto recipe = mod.lookupSymbol<aiir::acc::ReductionRecipeOp>(recipeName))
    return aiir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());

  aiir::OpBuilder::InsertionGuard guard(builder);
  bool allConstantBound = fir::acc::areAllBoundsConstant(dataBoundOps);
  auto recipe = genRecipeOp<aiir::acc::ReductionRecipeOp>(
      builder, mod, recipeName, loc, ty, varInfo, dataBoundOps,
      allConstantBound, op);

  auto [dest, source] = genRecipeCombinerOrCopyRegion(
      builder, loc, ty, recipe.getCombinerRegion(), dataBoundOps,
      allConstantBound);
  llvm::SmallVector<aiir::Value> combinerBounds =
      getRecipeBounds(builder, loc, dataBoundOps,
                      recipe.getCombinerRegion().getArguments().drop_front(2));

  [[maybe_unused]] bool success = mappableTy.generateCombiner(
      builder, loc, dest, source, combinerBounds, op, fastMathAttr);
  assert(success && "failed to generate combiner");
  aiir::acc::YieldOp::create(builder, loc, dest);
  return aiir::SymbolRefAttr::get(builder.getContext(), recipe.getSymName());
}

aiir::Value fir::acc::getOriginalDef(aiir::Value value, bool stripDeclare) {
  aiir::Value currentValue = value;

  while (currentValue) {
    auto *definingOp = currentValue.getDefiningOp();
    if (!definingOp)
      break;

    if (auto convertOp = aiir::dyn_cast<fir::ConvertOp>(definingOp)) {
      currentValue = convertOp.getValue();
      continue;
    }

    if (auto viewLike = aiir::dyn_cast<aiir::ViewLikeOpInterface>(definingOp)) {
      currentValue = viewLike.getViewSource();
      continue;
    }

    if (stripDeclare) {
      if (auto declareOp = aiir::dyn_cast<hlfir::DeclareOp>(definingOp)) {
        currentValue = declareOp.getMemref();
        continue;
      }

      if (auto declareOp = aiir::dyn_cast<fir::DeclareOp>(definingOp)) {
        currentValue = declareOp.getMemref();
        continue;
      }
    }
    break;
  }

  return currentValue;
}
