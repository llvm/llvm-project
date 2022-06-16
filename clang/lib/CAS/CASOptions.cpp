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
#include "llvm/CAS/CASDB.h"

using namespace clang;
using namespace llvm::cas;

static std::shared_ptr<llvm::cas::CASDB>
createCAS(const CASConfiguration &Config, DiagnosticsEngine &Diags,
          bool CreateEmptyCASOnFailure) {
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
  return CreateEmptyCASOnFailure ? llvm::cas::createInMemoryCAS() : nullptr;
}

std::shared_ptr<llvm::cas::CASDB>
CASOptions::getOrCreateCAS(DiagnosticsEngine &Diags,
                           bool CreateEmptyCASOnFailure) const {
  if (Cache.Config.IsFrozen)
    return Cache.CAS;

  auto &CurrentConfig = static_cast<const CASConfiguration &>(*this);
  if (!Cache.CAS || CurrentConfig != Cache.Config) {
    Cache.Config = CurrentConfig;
    Cache.CAS = createCAS(Cache.Config, Diags, CreateEmptyCASOnFailure);
  }

  return Cache.CAS;
}

std::shared_ptr<llvm::cas::CASDB>
CASOptions::getOrCreateCASAndHideConfig(DiagnosticsEngine &Diags) {
  if (Cache.Config.IsFrozen)
    return Cache.CAS;

  std::shared_ptr<llvm::cas::CASDB> CAS = getOrCreateCAS(Diags);
  assert(CAS == Cache.CAS && "Expected CAS to be cached");

  // Freeze the CAS and wipe out the visible config to hide it from future
  // accesses. For example, future diagnostics cannot see this. Something that
  // needs direct access to the CAS configuration will need to be
  // scheduled/executed at a level that has access to the configuration.
  auto &CurrentConfig = static_cast<CASConfiguration &>(*this);
  CurrentConfig = CASConfiguration();
  CurrentConfig.IsFrozen = Cache.Config.IsFrozen = true;

  if (CAS) {
    // Set the CASPath to the hash schema, since that leaks through CASContext's
    // API and is observable.
    CurrentConfig.CASPath = CAS->getHashSchemaIdentifier().str();
  }

  return CAS;
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
