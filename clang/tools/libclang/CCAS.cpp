//===- CCAS.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-c/CAS.h"

#include "CASUtils.h"
#include "CXString.h"

#include "clang/Basic/LLVM.h"
#include "clang/CAS/CASOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::cas;

CXCASOptions clang_experimental_cas_Options_create(void) {
  return wrap(new CASOptions());
}

void clang_experimental_cas_Options_dispose(CXCASOptions Opts) {
  delete unwrap(Opts);
}

void clang_experimental_cas_Options_setOnDiskPath(CXCASOptions COpts,
                                                  const char *Path) {
  CASOptions &Opts = *unwrap(COpts);
  Opts.CASPath = Path;
}

void clang_experimental_cas_Options_setPluginPath(CXCASOptions COpts,
                                                  const char *Path) {
  CASOptions &Opts = *unwrap(COpts);
  Opts.PluginPath = Path;
}

void clang_experimental_cas_Options_setPluginOption(CXCASOptions COpts,
                                                    const char *Name,
                                                    const char *Value) {
  CASOptions &Opts = *unwrap(COpts);
  Opts.PluginOptions.emplace_back(Name, Value);
}

CXCASDatabases clang_experimental_cas_Databases_create(CXCASOptions COpts,
                                                       CXString *Error) {
  CASOptions &Opts = *unwrap(COpts);

  SmallString<128> DiagBuf;
  llvm::raw_svector_ostream OS(DiagBuf);
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagPrinter(OS, DiagOpts.get());
  DiagnosticsEngine Diags(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), DiagOpts.get(),
      &DiagPrinter, /*ShouldOwnClient=*/false);

  auto [CAS, Cache] = Opts.getOrCreateDatabases(Diags);
  if (!CAS || !Cache) {
    if (Error)
      *Error = cxstring::createDup(OS.str());
    return nullptr;
  }

  return wrap(new WrappedCASDatabases{Opts, std::move(CAS), std::move(Cache)});
}

void clang_experimental_cas_Databases_dispose(CXCASDatabases CDBs) {
  delete unwrap(CDBs);
}

void clang_experimental_cas_ObjectStore_dispose(CXCASObjectStore CAS) {
  delete unwrap(CAS);
}
void clang_experimental_cas_ActionCache_dispose(CXCASActionCache Cache) {
  delete unwrap(Cache);
}

CXCASObjectStore
clang_experimental_cas_OnDiskObjectStore_create(const char *Path,
                                                CXString *Error) {
  auto CAS = llvm::cas::createOnDiskCAS(Path);
  if (!CAS) {
    if (Error)
      *Error = cxstring::createDup(llvm::toString(CAS.takeError()));
    return nullptr;
  }
  return wrap(new WrappedObjectStore{std::move(*CAS), Path});
}

CXCASActionCache
clang_experimental_cas_OnDiskActionCache_create(const char *Path,
                                                CXString *Error) {
  auto Cache = llvm::cas::createOnDiskActionCache(Path);
  if (!Cache) {
    if (Error)
      *Error = cxstring::createDup(llvm::toString(Cache.takeError()));
    return nullptr;
  }
  return wrap(new WrappedActionCache{std::move(*Cache), Path});
}
