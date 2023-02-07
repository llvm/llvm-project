//===- CASUtils.h - libclang CAS utilities --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_CASUTILS_H
#define LLVM_CLANG_TOOLS_LIBCLANG_CASUTILS_H

#include "clang-c/CAS.h"
#include "clang/Basic/LLVM.h"
#include "clang/CAS/CASOptions.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/CBindingWrapping.h"
#include <string>

namespace clang {
namespace cas {

struct WrappedCASDatabases {
  CASOptions CASOpts;
  std::shared_ptr<cas::ObjectStore> CAS;
  std::shared_ptr<cas::ActionCache> Cache;
};

struct WrappedObjectStore {
  std::shared_ptr<ObjectStore> CAS;
  std::string CASPath;
};

struct WrappedActionCache {
  std::shared_ptr<ActionCache> Cache;
  std::string CachePath;
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(CASOptions, CXCASOptions)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedCASDatabases, CXCASDatabases)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedObjectStore, CXCASObjectStore)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(WrappedActionCache, CXCASActionCache)

} // namespace cas
} // namespace clang

#endif // LLVM_CLANG_TOOLS_LIBCLANG_CASUTILS_H
