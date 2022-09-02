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
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"

using namespace clang;
using namespace llvm::cas;

static std::shared_ptr<llvm::cas::ObjectStore>
createObjectStore(const CASConfiguration &Config, DiagnosticsEngine &Diags) {
  if (Config.CASPath.empty())
    return llvm::cas::createInMemoryCAS();

  // Compute the path.
  SmallString<128> Storage;
  StringRef Path = Config.CASPath;
  if (Path == "auto") {
    llvm::cas::getDefaultOnDiskCASPath(Storage);
    Path = Storage;
  }

  // FIXME: Pass on the actual error from the CAS.
  if (auto MaybeCAS =
          llvm::expectedToOptional(llvm::cas::createOnDiskCAS(Path)))
    return std::move(*MaybeCAS);
  Diags.Report(diag::err_builtin_cas_cannot_be_initialized) << Path;
  return nullptr;
}

std::shared_ptr<llvm::cas::ObjectStore>
CASOptions::getOrCreateObjectStore(DiagnosticsEngine &Diags,
                                   bool CreateEmptyCASOnFailure) const {
  if (Cache.Config.IsFrozen)
    return Cache.CAS;

  initCache(Diags);
  if (Cache.CAS)
    return Cache.CAS;
  if (!CreateEmptyCASOnFailure)
    return nullptr;
  Cache.CAS = llvm::cas::createInMemoryCAS();
  return Cache.CAS;
}

void CASOptions::freezeConfig(DiagnosticsEngine &Diags) {
  if (Cache.Config.IsFrozen)
    return;

  // Make sure the cache is initialized.
  initCache(Diags);

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
  }
  if (Cache.AC)
    CurrentConfig.CachePath = "";
}

static std::shared_ptr<llvm::cas::ActionCache>
createCache(ObjectStore &CAS, const CASConfiguration &Config,
            DiagnosticsEngine &Diags) {
  if (Config.CachePath.empty())
    return llvm::cas::createInMemoryActionCache();

  // Compute the path.
  std::string Path = Config.CachePath;
  if (Path == "auto")
    Path = getDefaultOnDiskActionCachePath();

  // FIXME: Pass on the actual error from the CAS.
  if (auto MaybeCache =
          llvm::expectedToOptional(llvm::cas::createOnDiskActionCache(Path)))
    return std::move(*MaybeCache);
  Diags.Report(diag::err_builtin_actioncache_cannot_be_initialized) << Path;
  return nullptr;
}

std::shared_ptr<llvm::cas::ActionCache>
CASOptions::getOrCreateActionCache(DiagnosticsEngine &Diags,
                                   bool CreateEmptyOnFailure) const {
  if (Cache.Config.IsFrozen)
    return Cache.AC;

  initCache(Diags);
  if (Cache.AC)
    return Cache.AC;
  if (!CreateEmptyOnFailure)
    return nullptr;

  Cache.CAS = Cache.CAS ? Cache.CAS : llvm::cas::createInMemoryCAS();
  return llvm::cas::createInMemoryActionCache();
}

void CASOptions::ensurePersistentCAS() {
  assert(!IsFrozen && "Expected to check for a persistent CAS before freezing");
  switch (getKind()) {
  case UnknownCAS:
      llvm_unreachable("Cannot ensure persistent CAS if it's unknown / frozen");
  case InMemoryCAS:
    CASPath = "auto";
    CachePath = "auto";
    break;
  case OnDiskCAS:
    break;
  }
}

void CASOptions::initCache(DiagnosticsEngine &Diags) const {
  auto &CurrentConfig = static_cast<const CASConfiguration &>(*this);
  if (CurrentConfig == Cache.Config && Cache.CAS && Cache.AC)
    return;

  Cache.Config = CurrentConfig;
  Cache.CAS = createObjectStore(Cache.Config, Diags);
  Cache.AC = createCache(*Cache.CAS, Cache.Config, Diags);
}
