//===- SerializationFormatRegistry.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include <memory>

using namespace clang;
using namespace ssaf;

// FIXME: LLVM_INSTANTIATE_REGISTRY can't be used here because it drops extra
// type parameters.
template class CLANG_EXPORT_TEMPLATE
    llvm::Registry<clang::ssaf::SerializationFormat,
                   llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>>;

bool ssaf::isFormatRegistered(llvm::StringRef FormatName) {
  for (const auto &Entry : SerializationFormatRegistry::entries())
    if (Entry.getName() == FormatName)
      return true;
  return false;
}

std::unique_ptr<SerializationFormat>
ssaf::makeFormat(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                 llvm::StringRef FormatName) {
  for (const auto &Entry : SerializationFormatRegistry::entries())
    if (Entry.getName() == FormatName)
      return Entry.instantiate(std::move(FS));
  assert(false && "Unknown SerializationFormat name");
  return nullptr;
}
