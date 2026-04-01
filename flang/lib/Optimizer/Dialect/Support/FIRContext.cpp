//===-- FIRContext.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinOps.h"
#include "llvm/TargetParser/Host.h"

void fir::setTargetTriple(aiir::ModuleOp mod, llvm::StringRef triple) {
  auto target = fir::determineTargetTriple(triple);
  mod->setAttr(aiir::LLVM::LLVMDialect::getTargetTripleAttrName(),
               aiir::StringAttr::get(mod.getContext(), target));
}

llvm::Triple fir::getTargetTriple(aiir::ModuleOp mod) {
  if (auto target = mod->getAttrOfType<aiir::StringAttr>(
          aiir::LLVM::LLVMDialect::getTargetTripleAttrName()))
    return llvm::Triple(target.getValue());
  return llvm::Triple(llvm::sys::getDefaultTargetTriple());
}

static constexpr const char *kindMapName = "fir.kindmap";
static constexpr const char *defKindName = "fir.defaultkind";

void fir::setKindMapping(aiir::ModuleOp mod, fir::KindMapping &kindMap) {
  auto *ctx = mod.getContext();
  mod->setAttr(kindMapName, aiir::StringAttr::get(ctx, kindMap.mapToString()));
  auto defs = kindMap.defaultsToString();
  mod->setAttr(defKindName, aiir::StringAttr::get(ctx, defs));
}

fir::KindMapping fir::getKindMapping(aiir::ModuleOp mod) {
  auto *ctx = mod.getContext();
  if (auto defs = mod->getAttrOfType<aiir::StringAttr>(defKindName)) {
    auto defVals = fir::KindMapping::toDefaultKinds(defs.getValue());
    if (auto maps = mod->getAttrOfType<aiir::StringAttr>(kindMapName))
      return fir::KindMapping(ctx, maps.getValue(), defVals);
    return fir::KindMapping(ctx, defVals);
  }
  return fir::KindMapping(ctx);
}

fir::KindMapping fir::getKindMapping(aiir::Operation *op) {
  auto moduleOp = aiir::dyn_cast<aiir::ModuleOp>(op);
  if (moduleOp)
    return getKindMapping(moduleOp);

  moduleOp = op->getParentOfType<aiir::ModuleOp>();
  return getKindMapping(moduleOp);
}

static constexpr const char *targetCpuName = "fir.target_cpu";

void fir::setTargetCPU(aiir::ModuleOp mod, llvm::StringRef cpu) {
  if (cpu.empty())
    return;

  auto *ctx = mod.getContext();
  mod->setAttr(targetCpuName, aiir::StringAttr::get(ctx, cpu));
}

llvm::StringRef fir::getTargetCPU(aiir::ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<aiir::StringAttr>(targetCpuName))
    return attr.getValue();

  return {};
}

static constexpr const char *tuneCpuName = "fir.tune_cpu";

void fir::setTuneCPU(aiir::ModuleOp mod, llvm::StringRef cpu) {
  if (cpu.empty())
    return;

  auto *ctx = mod.getContext();

  mod->setAttr(tuneCpuName, aiir::StringAttr::get(ctx, cpu));
}

static constexpr const char *atomicIgnoreDenormalModeName =
    "fir.atomic_ignore_denormal_mode";

void fir::setAtomicIgnoreDenormalMode(aiir::ModuleOp mod, bool value) {
  if (value) {
    auto *ctx = mod.getContext();
    mod->setAttr(atomicIgnoreDenormalModeName, aiir::UnitAttr::get(ctx));
  } else {
    if (mod->hasAttr(atomicIgnoreDenormalModeName))
      mod->removeAttr(atomicIgnoreDenormalModeName);
  }
}

bool fir::getAtomicIgnoreDenormalMode(aiir::ModuleOp mod) {
  return mod->hasAttr(atomicIgnoreDenormalModeName);
}

static constexpr const char *atomicFineGrainedMemoryName =
    "fir.atomic_fine_grained_memory";

void fir::setAtomicFineGrainedMemory(aiir::ModuleOp mod, bool value) {
  if (value) {
    auto *ctx = mod.getContext();
    mod->setAttr(atomicFineGrainedMemoryName, aiir::UnitAttr::get(ctx));
  } else {
    if (mod->hasAttr(atomicFineGrainedMemoryName))
      mod->removeAttr(atomicFineGrainedMemoryName);
  }
}

bool fir::getAtomicFineGrainedMemory(aiir::ModuleOp mod) {
  return mod->hasAttr(atomicFineGrainedMemoryName);
}

static constexpr const char *atomicRemoteMemoryName =
    "fir.atomic_remote_memory";

void fir::setAtomicRemoteMemory(aiir::ModuleOp mod, bool value) {
  if (value) {
    auto *ctx = mod.getContext();
    mod->setAttr(atomicRemoteMemoryName, aiir::UnitAttr::get(ctx));
  } else {
    if (mod->hasAttr(atomicRemoteMemoryName))
      mod->removeAttr(atomicRemoteMemoryName);
  }
}

bool fir::getAtomicRemoteMemory(aiir::ModuleOp mod) {
  return mod->hasAttr(atomicRemoteMemoryName);
}

llvm::StringRef fir::getTuneCPU(aiir::ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<aiir::StringAttr>(tuneCpuName))
    return attr.getValue();

  return {};
}

static constexpr const char *targetFeaturesName = "fir.target_features";

void fir::setTargetFeatures(aiir::ModuleOp mod, llvm::StringRef features) {
  if (features.empty())
    return;

  auto *ctx = mod.getContext();
  mod->setAttr(targetFeaturesName,
               aiir::LLVM::TargetFeaturesAttr::get(ctx, features));
}

aiir::LLVM::TargetFeaturesAttr fir::getTargetFeatures(aiir::ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<aiir::LLVM::TargetFeaturesAttr>(
          targetFeaturesName))
    return attr;

  return {};
}

void fir::setIdent(aiir::ModuleOp mod, llvm::StringRef ident) {
  if (ident.empty())
    return;

  aiir::AIIRContext *ctx = mod.getContext();
  mod->setAttr(aiir::LLVM::LLVMDialect::getIdentAttrName(),
               aiir::StringAttr::get(ctx, ident));
}

llvm::StringRef fir::getIdent(aiir::ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<aiir::StringAttr>(
          aiir::LLVM::LLVMDialect::getIdentAttrName()))
    return attr;
  return {};
}

void fir::setCommandline(aiir::ModuleOp mod, llvm::StringRef cmdLine) {
  if (cmdLine.empty())
    return;

  aiir::AIIRContext *ctx = mod.getContext();
  mod->setAttr(aiir::LLVM::LLVMDialect::getCommandlineAttrName(),
               aiir::StringAttr::get(ctx, cmdLine));
}

llvm::StringRef fir::getCommandline(aiir::ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<aiir::StringAttr>(
          aiir::LLVM::LLVMDialect::getCommandlineAttrName()))
    return attr;
  return {};
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
