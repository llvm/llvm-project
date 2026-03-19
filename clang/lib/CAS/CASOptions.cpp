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
CASOptions::createDatabases(DiagnosticsEngine &Diags,
                            bool CreateEmptyDBsOnFailure) const {
  auto DBs = CASConfiguration::createDatabases();
  if (DBs)
    return std::move(*DBs);

  Diags.Report(diag::err_cas_cannot_be_initialized)
      << toString(DBs.takeError());

  if (CreateEmptyDBsOnFailure)
    return {llvm::cas::createInMemoryCAS(),
            llvm::cas::createInMemoryActionCache()};
  return {nullptr, nullptr};
}

void CASOptions::ensurePersistentCAS() {
  switch (getKind()) {
  case InMemoryCAS:
    CASPath = "auto";
    break;
  case OnDiskCAS:
    break;
  }
}
