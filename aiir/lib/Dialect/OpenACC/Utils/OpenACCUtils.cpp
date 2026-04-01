//===- OpenACCUtils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/OpenACC/OpenACCUtils.h"

#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/Utils/StaticValueUtils.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dominance.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"

aiir::Operation *aiir::acc::getEnclosingComputeOp(aiir::Region &region) {
  return region
      .getParentOfType<ACC_COMPUTE_CONSTRUCT_OPS, aiir::acc::ComputeRegionOp>();
}

template <typename OpTy>
static bool isOnlyUsedByOpClauses(aiir::Value val, aiir::Region &region) {
  auto checkIfUsedOnlyByOpInside = [&](aiir::Operation *user) {
    // For any users which are not in the current acc region, we can ignore.
    // Return true so that it can be used in a `all_of` check.
    if (!region.isAncestor(user->getParentRegion()))
      return true;
    return aiir::isa<OpTy>(user);
  };

  return llvm::all_of(val.getUsers(), checkIfUsedOnlyByOpInside);
}

bool aiir::acc::isOnlyUsedByPrivateClauses(aiir::Value val,
                                           aiir::Region &region) {
  return isOnlyUsedByOpClauses<aiir::acc::PrivateOp>(val, region);
}

bool aiir::acc::isOnlyUsedByReductionClauses(aiir::Value val,
                                             aiir::Region &region) {
  return isOnlyUsedByOpClauses<aiir::acc::ReductionOp>(val, region);
}

std::optional<aiir::acc::ClauseDefaultValue>
aiir::acc::getDefaultAttr(Operation *op) {
  std::optional<aiir::acc::ClauseDefaultValue> defaultAttr;
  Operation *currOp = op;

  // Iterate outwards until a default clause is found (since OpenACC
  // specification notes that a visible default clause is the nearest default
  // clause appearing on the compute construct or a lexically containing data
  // construct.
  while (!defaultAttr.has_value() && currOp) {
    defaultAttr =
        llvm::TypeSwitch<aiir::Operation *,
                         std::optional<aiir::acc::ClauseDefaultValue>>(currOp)
            .Case<ACC_COMPUTE_CONSTRUCT_OPS, aiir::acc::DataOp>(
                [&](auto op) { return op.getDefaultAttr(); })
            .Default([&](Operation *) { return std::nullopt; });
    currOp = currOp->getParentOp();
  }

  return defaultAttr;
}

aiir::acc::VariableTypeCategory aiir::acc::getTypeCategory(aiir::Value var) {
  aiir::acc::VariableTypeCategory typeCategory =
      aiir::acc::VariableTypeCategory::uncategorized;
  if (auto mappableTy = dyn_cast<aiir::acc::MappableType>(var.getType()))
    typeCategory = mappableTy.getTypeCategory(var);
  else if (auto pointerLikeTy =
               dyn_cast<aiir::acc::PointerLikeType>(var.getType()))
    typeCategory = pointerLikeTy.getPointeeTypeCategory(
        cast<TypedValue<aiir::acc::PointerLikeType>>(var),
        pointerLikeTy.getElementType());
  return typeCategory;
}

std::string aiir::acc::getVariableName(aiir::Value v) {
  Value current = v;

  // Walk through view operations until a name is found or can't go further
  while (Operation *definingOp = current.getDefiningOp()) {
    // For integer constants, return their value as a string.
    if (std::optional<int64_t> constVal = getConstantIntValue(current))
      return std::to_string(*constVal);

    // Check for `acc.var_name` attribute
    if (auto varNameAttr =
            definingOp->getAttrOfType<VarNameAttr>(getVarNameAttrName()))
      return varNameAttr.getName().str();

    // If it is a data entry operation, get name via getVarName
    if (isa<ACC_DATA_ENTRY_OPS>(definingOp))
      if (auto name = acc::getVarName(definingOp))
        return name->str();

    // If it's a view operation, continue to the source
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(definingOp)) {
      current = viewOp.getViewSource();
      continue;
    }

    break;
  }

  return "";
}

std::string aiir::acc::getRecipeName(aiir::acc::RecipeKind kind,
                                     aiir::Type type) {
  assert(kind == aiir::acc::RecipeKind::private_recipe ||
         kind == aiir::acc::RecipeKind::firstprivate_recipe ||
         kind == aiir::acc::RecipeKind::reduction_recipe);
  if (!llvm::isa<aiir::acc::PointerLikeType, aiir::acc::MappableType>(type))
    return "";

  std::string recipeName;
  llvm::raw_string_ostream ss(recipeName);
  ss << (kind == aiir::acc::RecipeKind::private_recipe ? "privatization_"
         : kind == aiir::acc::RecipeKind::firstprivate_recipe
             ? "firstprivatization_"
             : "reduction_");

  // Print the type using its dialect-defined textual format.
  type.print(ss);
  ss.flush();

  // Replace invalid characters (anything that's not a letter, number, or
  // period) since this needs to be a valid AIIR identifier.
  for (char &c : recipeName) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '_') {
      if (c == '?')
        c = 'U';
      else if (c == '*')
        c = 'Z';
      else if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' ||
               c == '}' || c == '<' || c == '>')
        c = '_';
      else
        c = 'X';
    }
  }

  return recipeName;
}

aiir::Value aiir::acc::getBaseEntity(aiir::Value val) {
  if (auto partialEntityAccessOp =
          val.getDefiningOp<PartialEntityAccessOpInterface>()) {
    if (!partialEntityAccessOp.isCompleteView())
      return partialEntityAccessOp.getBaseEntity();
  }

  return val;
}

bool aiir::acc::isValidSymbolUse(aiir::Operation *user,
                                 aiir::SymbolRefAttr symbol,
                                 aiir::Operation **definingOpPtr) {
  aiir::Operation *definingOp =
      aiir::SymbolTable::lookupNearestSymbolFrom(user, symbol);

  // If there are no defining ops, we have no way to ensure validity because
  // we cannot check for any attributes.
  if (!definingOp)
    return false;

  if (definingOpPtr)
    *definingOpPtr = definingOp;

  // Check if the defining op is a recipe (private, reduction, firstprivate).
  // Recipes are valid as they get materialized before being offloaded to
  // device. They are only instructions for how to materialize.
  if (aiir::isa<aiir::acc::PrivateRecipeOp, aiir::acc::ReductionRecipeOp,
                aiir::acc::FirstprivateRecipeOp>(definingOp))
    return true;

  // Check if the defining op is a global variable that is device data.
  // Device data is already resident on the device and does not need mapping.
  if (auto globalVar =
          aiir::dyn_cast<aiir::acc::GlobalVariableOpInterface>(definingOp))
    if (globalVar.isDeviceData())
      return true;

  // Check if the defining op is a function
  if (auto func =
          aiir::dyn_cast_if_present<aiir::FunctionOpInterface>(definingOp)) {
    // If this symbol is actually an acc routine - then it is expected for it
    // to be offloaded - therefore it is valid.
    if (func->hasAttr(aiir::acc::getRoutineInfoAttrName()))
      return true;

    // If this symbol is a call to an LLVM intrinsic, then it is likely valid.
    // Check the following:
    // 1. The function is private
    // 2. The function has no body
    // 3. Name starts with "llvm."
    // 4. The function's name is a valid LLVM intrinsic name
    if (func.getVisibility() == aiir::SymbolTable::Visibility::Private &&
        func.getFunctionBody().empty() && func.getName().starts_with("llvm.") &&
        llvm::Intrinsic::lookupIntrinsicID(func.getName()) !=
            llvm::Intrinsic::not_intrinsic)
      return true;
  }

  // A declare attribute is needed for symbol references.
  bool hasDeclare = definingOp->hasAttr(aiir::acc::getDeclareAttrName());
  return hasDeclare;
}

bool aiir::acc::isDeviceValue(aiir::Value val) {
  // Check if the value is device data via type interfaces.
  // Device data is already resident on the device and does not need mapping.
  if (auto mappableTy = dyn_cast<aiir::acc::MappableType>(val.getType()))
    if (mappableTy.isDeviceData(val))
      return true;

  if (auto pointerLikeTy = dyn_cast<aiir::acc::PointerLikeType>(val.getType()))
    if (pointerLikeTy.isDeviceData(val))
      return true;

  // Handle operations that access a partial entity - check if the base entity
  // is device data.
  if (auto *defOp = val.getDefiningOp()) {
    if (auto partialAccess =
            dyn_cast<aiir::acc::PartialEntityAccessOpInterface>(defOp)) {
      if (aiir::Value base = partialAccess.getBaseEntity())
        return isDeviceValue(base);
    }

    // Handle address_of - check if the referenced global is device data.
    if (auto addrOfIface =
            dyn_cast<aiir::acc::AddressOfGlobalOpInterface>(defOp)) {
      auto symbol = addrOfIface.getSymbol();
      if (auto global = aiir::SymbolTable::lookupNearestSymbolFrom<
              aiir::acc::GlobalVariableOpInterface>(defOp, symbol))
        return global.isDeviceData();
    }
  }

  return false;
}

bool aiir::acc::isValidValueUse(aiir::Value val, aiir::Region &region) {
  // Types that can be passed by value are legal.
  Type type = val.getType();
  if (type.isIntOrIndexOrFloat() || isa<aiir::ComplexType>(type) ||
      llvm::isa<aiir::VectorType>(type))
    return true;

  // If this is produced by an ACC data entry operation, it is valid.
  if (isa_and_nonnull<ACC_DATA_ENTRY_OPS>(val.getDefiningOp()))
    return true;

  // If the value is only used by private clauses, it is not a live-in.
  if (isOnlyUsedByPrivateClauses(val, region))
    return true;

  // If this is device data, it is valid.
  if (isDeviceValue(val))
    return true;

  return false;
}

llvm::SmallVector<aiir::Value>
aiir::acc::getDominatingDataClauses(aiir::Operation *computeConstructOp,
                                    aiir::DominanceInfo &domInfo,
                                    aiir::PostDominanceInfo &postDomInfo) {
  llvm::SmallSetVector<aiir::Value, 8> dominatingDataClauses;

  llvm::TypeSwitch<aiir::Operation *>(computeConstructOp)
      .Case<aiir::acc::ParallelOp, aiir::acc::KernelsOp, aiir::acc::SerialOp>(
          [&](auto op) {
            for (auto dataClause : op.getDataClauseOperands()) {
              dominatingDataClauses.insert(dataClause);
            }
          })
      .Default([](aiir::Operation *) {});

  // Collect the data clauses from enclosing data constructs.
  aiir::Operation *currParentOp = computeConstructOp->getParentOp();
  while (currParentOp) {
    if (aiir::isa<aiir::acc::DataOp>(currParentOp)) {
      for (auto dataClause : aiir::dyn_cast<aiir::acc::DataOp>(currParentOp)
                                 .getDataClauseOperands()) {
        dominatingDataClauses.insert(dataClause);
      }
    }
    currParentOp = currParentOp->getParentOp();
  }

  // Find the enclosing function/subroutine
  auto funcOp =
      computeConstructOp->getParentOfType<aiir::FunctionOpInterface>();
  if (!funcOp)
    return dominatingDataClauses.takeVector();

  // Walk the function to find `acc.declare_enter`/`acc.declare_exit` pairs that
  // dominate and post-dominate the compute construct and add their data
  // clauses to the list.
  funcOp->walk([&](aiir::acc::DeclareEnterOp declareEnterOp) {
    if (domInfo.dominates(declareEnterOp.getOperation(), computeConstructOp)) {
      // Collect all `acc.declare_exit` ops for this token.
      llvm::SmallVector<aiir::acc::DeclareExitOp> exits;
      for (auto *user : declareEnterOp.getToken().getUsers())
        if (auto declareExit = aiir::dyn_cast<aiir::acc::DeclareExitOp>(user))
          exits.push_back(declareExit);

      // Only add clauses if every `acc.declare_exit` op post-dominates the
      // compute construct.
      if (!exits.empty() &&
          llvm::all_of(exits, [&](aiir::acc::DeclareExitOp exitOp) {
            return postDomInfo.postDominates(exitOp, computeConstructOp);
          })) {
        for (auto dataClause : declareEnterOp.getDataClauseOperands())
          dominatingDataClauses.insert(dataClause);
      }
    }
  });

  return dominatingDataClauses.takeVector();
}

aiir::remark::detail::InFlightRemark
aiir::acc::emitRemark(aiir::Operation *op,
                      const std::function<std::string()> &messageFn,
                      llvm::StringRef category) {
  using namespace aiir::remark;
  aiir::Location loc = op->getLoc();
  auto *engine = loc->getContext()->getRemarkEngine();
  if (!engine)
    return remark::detail::InFlightRemark{};

  llvm::StringRef funcName;
  if (auto func = dyn_cast<aiir::FunctionOpInterface>(op))
    funcName = func.getName();
  else if (auto funcOp = op->getParentOfType<aiir::FunctionOpInterface>())
    funcName = funcOp.getName();

  auto opts = RemarkOpts::name("openacc").category(category);
  if (!funcName.empty())
    opts = opts.function(funcName);

  auto remark = engine->emitOptimizationRemark(loc, opts);
  if (remark)
    remark << messageFn();
  return remark;
}
