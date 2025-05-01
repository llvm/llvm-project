//===- CASOptions.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CAS/CASOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace llvm::cas;

std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
          std::shared_ptr<llvm::cas::ActionCache>>
CASOptions::getOrCreateDatabases(DiagnosticsEngine &Diags,
                                 bool CreateEmptyDBsOnFailure) const {
  if (Cache.Config.IsFrozen)
    return {Cache.CAS, Cache.AC};

  if (auto E = initCache())
    Diags.Report(diag::err_cas_cannot_be_initialized) << toString(std::move(E));

  if (!Cache.CAS && CreateEmptyDBsOnFailure)
    Cache.CAS = llvm::cas::createInMemoryCAS();
  if (!Cache.AC && CreateEmptyDBsOnFailure)
    Cache.AC = llvm::cas::createInMemoryActionCache();
  return {Cache.CAS, Cache.AC};
}

llvm::Expected<std::pair<std::shared_ptr<llvm::cas::ObjectStore>,
                         std::shared_ptr<llvm::cas::ActionCache>>>
CASOptions::getOrCreateDatabases() const {
  if (auto E = initCache())
    return std::move(E);
  return std::pair{Cache.CAS, Cache.AC};
}

void CASOptions::freezeConfig(DiagnosticsEngine &Diags) {
  if (Cache.Config.IsFrozen)
    return;

  // Make sure the cache is initialized.
  if (auto E = initCache())
    Diags.Report(diag::err_cas_cannot_be_initialized) << toString(std::move(E));

  // Freeze the CAS and wipe out the visible config to hide it from future
  // accesses. For example, future diagnostics cannot see this. Something that
  // needs direct access to the CAS configuration will need to be
  // scheduled/executed at a level that has access to the configuration.
  auto &CurrentConfig = static_cast<CASConfiguration &>(*this);
  CurrentConfig = CASConfiguration();
  CurrentConfig.IsFrozen = Cache.Config.IsFrozen = true;

  if (Cache.CAS) {
    // Set the CASPath to the hash schema, since that leaks through CASContext's
    // API and is observable.
    CurrentConfig.CASPath =
        Cache.CAS->getContext().getHashSchemaIdentifier().str();
    CurrentConfig.PluginPath.clear();
    CurrentConfig.PluginOptions.clear();
  }
}

void CASOptions::ensurePersistentCAS() {
  assert(!IsFrozen && "Expected to check for a persistent CAS before freezing");
  switch (getKind()) {
  case UnknownCAS:
    llvm_unreachable("Cannot ensure persistent CAS if it's unknown / frozen");
  case InMemoryCAS:
    CASPath = "auto";
    break;
  case OnDiskCAS:
    break;
  }
}

llvm::Error CASOptions::initCache() const {
  auto &CurrentConfig = static_cast<const CASConfiguration &>(*this);
  if (CurrentConfig == Cache.Config && Cache.CAS && Cache.AC)
    return llvm::Error::success();

  Cache.Config = CurrentConfig;
  StringRef CASPath = Cache.Config.CASPath;

  if (!PluginPath.empty()) {
    std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>> DBs;
    if (llvm::Error E =
            createPluginCASDatabases(PluginPath, CASPath, PluginOptions)
                .moveInto(DBs)) {
      return E;
    }
    std::tie(Cache.CAS, Cache.AC) = std::move(DBs);
    return llvm::Error::success();
  }

  if (CASPath.empty()) {
    Cache.CAS = llvm::cas::createInMemoryCAS();
    Cache.AC = llvm::cas::createInMemoryActionCache();
    return llvm::Error::success();
  }

  SmallString<256> PathBuf;
  getResolvedCASPath(PathBuf);
  if (CASPath == "auto") {
    getDefaultOnDiskCASPath(PathBuf);
    CASPath = PathBuf;
  }
  std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>> DBs;
  if (llvm::Error E = createOnDiskUnifiedCASDatabases(CASPath).moveInto(DBs))
    return E;

  std::tie(Cache.CAS, Cache.AC) = std::move(DBs);
  return llvm::Error::success();
}

void CASOptions::getResolvedCASPath(SmallVectorImpl<char> &Result) const {
  if (CASPath == "auto") {
    getDefaultOnDiskCASPath(Result);
  } else {
    Result.assign(CASPath.begin(), CASPath.end());
  }
}
