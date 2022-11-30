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
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"

using namespace clang;
using namespace clang::cas;

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
