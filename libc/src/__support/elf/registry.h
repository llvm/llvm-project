//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Registry of modules loaded at runtime
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_REGISTRY_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_REGISTRY_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/elf/bind.h"
#include "src/__support/elf/load_module.h"
#include "src/__support/elf/module.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// How many objects may be open at once. A fixed table keeps the registry
// usable from contexts where the allocator is not available yet.
constexpr size_t MAX_LOADED_MODULES = 64;

struct RegistryEntry {
  LoadedModule loaded;
  size_t references = 0;
  bool in_use = false;
  // Set for objects the process was started with, which are never unloaded.
  bool permanent = false;
};

// The set of objects currently loaded, in the order they were loaded, which
// is the order symbol lookup walks them in.
class Registry {
public:
  LIBC_INLINE constexpr Registry() = default;

  LIBC_INLINE size_t count() const { return count_; }

  LIBC_INLINE RegistryEntry *begin() { return entries_; }
  LIBC_INLINE RegistryEntry *at(size_t index) {
    return index < count_ ? &entries_[index] : nullptr;
  }

  // Finds an already loaded object by the name it was opened under, so that
  // opening the same object twice shares one mapping.
  LIBC_INLINE RegistryEntry *find(const char *name) {
    for (size_t i = 0; i < count_; ++i) {
      if (!entries_[i].in_use)
        continue;
      const char *have = entries_[i].loaded.module.name();
      if (have != nullptr && equal(have, name))
        return &entries_[i];
    }
    return nullptr;
  }

  LIBC_INLINE RegistryEntry *add(const LoadedModule &loaded) {
    for (size_t i = 0; i < MAX_LOADED_MODULES; ++i) {
      if (entries_[i].in_use)
        continue;
      entries_[i].loaded = loaded;
      entries_[i].references = 1;
      entries_[i].in_use = true;
      entries_[i].permanent = false;
      if (i >= count_)
        count_ = i + 1;
      return &entries_[i];
    }
    return nullptr;
  }

  LIBC_INLINE void remove(RegistryEntry *entry) {
    entry->in_use = false;
    entry->references = 0;
  }

  // Builds the search order over everything currently loaded. The caller
  // supplies the storage so this stays allocation free.
  LIBC_INLINE size_t collect(Module *out, size_t capacity) {
    size_t written = 0;
    for (size_t i = 0; i < count_ && written < capacity; ++i)
      if (entries_[i].in_use)
        out[written++] = entries_[i].loaded.module;
    return written;
  }

private:
  LIBC_INLINE static bool equal(const char *a, const char *b) {
    for (; *a == *b; ++a, ++b)
      if (*a == '\0')
        return true;
    return false;
  }

  RegistryEntry entries_[MAX_LOADED_MODULES] = {};
  size_t count_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_REGISTRY_H
