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
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace fir {
namespace acc {

std::string getVariableName(Value v, bool preferDemangledName) {
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
    return acc::getVariableName(v);

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

bool areAllBoundsConstant(llvm::ArrayRef<Value> bounds) {
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

std::string getRecipeName(mlir::acc::RecipeKind kind, Type type, Value var,
                          llvm::ArrayRef<Value> bounds,
                          mlir::acc::ReductionOperator reductionOp) {
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

  auto kindMap = var && var.getDefiningOp()
                     ? fir::getKindMapping(var.getDefiningOp())
                     : fir::KindMapping(type.getContext());
  return fir::getTypeAsString(type, kindMap, prefixOS.str());
}

} // namespace acc
} // namespace fir
