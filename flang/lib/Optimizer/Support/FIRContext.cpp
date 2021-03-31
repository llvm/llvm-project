//===-- FIRContext.cpp ----------------------------------------------------===//
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

#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Host.h"

static constexpr const char *tripleName = "fir.triple";

void fir::setTargetTriple(mlir::ModuleOp mod, llvm::StringRef triple) {
  auto target = fir::determineTargetTriple(triple);
  mod->setAttr(tripleName, mlir::StringAttr::get(mod.getContext(), target));
}

llvm::Triple fir::getTargetTriple(mlir::ModuleOp mod) {
  if (auto target = mod->getAttrOfType<mlir::StringAttr>(tripleName))
    return llvm::Triple(target.getValue());
  return llvm::Triple(llvm::sys::getDefaultTargetTriple());
}

static constexpr const char *kindMapName = "fir.kindmap";
static constexpr const char *defKindName = "fir.defaultkind";

void fir::setKindMapping(mlir::ModuleOp mod, fir::KindMapping &kindMap) {
  auto ctx = mod.getContext();
  mod->setAttr(kindMapName, mlir::StringAttr::get(ctx, kindMap.mapToString()));
  auto defs = kindMap.defaultsToString();
  mod->setAttr(defKindName, mlir::StringAttr::get(ctx, defs));
}

fir::KindMapping fir::getKindMapping(mlir::ModuleOp mod) {
  auto ctx = mod.getContext();
  if (auto defs = mod->getAttrOfType<mlir::StringAttr>(defKindName)) {
    auto defVals = fir::KindMapping::toDefaultKinds(defs.getValue());
    if (auto maps = mod->getAttrOfType<mlir::StringAttr>(kindMapName))
      return fir::KindMapping(ctx, maps.getValue(), defVals);
    return fir::KindMapping(ctx, defVals);
  }
  return fir::KindMapping(ctx);
}

std::string fir::determineTargetTriple(llvm::StringRef triple) {
  // Treat "" or "default" as stand-ins for the default machine.
  if (triple.empty() || triple == "default")
    return llvm::sys::getDefaultTargetTriple();
  // Treat "native" as stand-in for the host machine.
  if (triple == "native")
    return llvm::sys::getProcessTriple();
  // TODO: normalize the triple?
  return triple.str();
}
