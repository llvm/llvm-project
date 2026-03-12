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

LLVM_INSTANTIATE_REGISTRY(SerializationFormatRegistry)

bool ssaf::isFormatRegistered(llvm::StringRef FormatName) {
  for (const auto &Entry : SerializationFormatRegistry::entries())
    if (Entry.getName() == FormatName)
      return true;
  return false;
}

std::unique_ptr<SerializationFormat>
ssaf::makeFormat(llvm::StringRef FormatName) {
  for (const auto &Entry : SerializationFormatRegistry::entries())
    if (Entry.getName() == FormatName)
      return Entry.instantiate();
  assert(false && "Unknown SerializationFormat name");
  return nullptr;
}
