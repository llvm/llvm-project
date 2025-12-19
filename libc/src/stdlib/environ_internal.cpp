//===-- Implementation of internal environment utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "environ_internal.h"
#include "config/app.h"
#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/func/realloc.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Minimum initial capacity for the environment array when first allocated.
// This avoids frequent reallocations for small environments.
constexpr size_t MIN_ENVIRON_CAPACITY = 32;

// Growth factor for environment array capacity when expanding.
// When capacity is exceeded, new_capacity = old_capacity *
// ENVIRON_GROWTH_FACTOR.
constexpr size_t ENVIRON_GROWTH_FACTOR = 2;

char **EnvironmentManager::get_array() {
  if (is_ours)
    return storage;
  return reinterpret_cast<char **>(LIBC_NAMESPACE::app.env_ptr);
}

void EnvironmentManager::init() {
  // Count entries in the startup environ
  char **env_ptr = reinterpret_cast<char **>(LIBC_NAMESPACE::app.env_ptr);
  if (!env_ptr)
    return;

  size_t count = 0;
  for (char **env = env_ptr; *env != nullptr; env++)
    count++;

  size = count;
}

int EnvironmentManager::find_var(cpp::string_view name) {
  char **env_array = get_array();
  if (!env_array)
    return -1;

  for (size_t i = 0; i < size; i++) {
    cpp::string_view current(env_array[i]);
    if (!current.starts_with(name))
      continue;

    // Check that name is followed by '='
    if (current.size() > name.size() && current[name.size()] == '=')
      return static_cast<int>(i);
  }

  return -1;
}

bool EnvironmentManager::ensure_capacity(size_t needed) {
  // If we're still using the startup environ, we need to copy it
  if (!is_ours) {
    char **old_env = reinterpret_cast<char **>(LIBC_NAMESPACE::app.env_ptr);

    // Allocate new array with room to grow
    size_t new_capacity = needed < MIN_ENVIRON_CAPACITY
                              ? MIN_ENVIRON_CAPACITY
                              : needed * ENVIRON_GROWTH_FACTOR;
    char **new_storage =
        reinterpret_cast<char **>(malloc(sizeof(char *) * (new_capacity + 1)));
    if (!new_storage)
      return false;

    // Allocate ownership tracking array
    EnvStringOwnership *new_ownership = reinterpret_cast<EnvStringOwnership *>(
        malloc(sizeof(EnvStringOwnership) * (new_capacity + 1)));
    if (!new_ownership) {
      free(new_storage);
      return false;
    }

    // Copy existing pointers (we don't own the strings yet, so just copy
    // pointers)
    if (old_env) {
      for (size_t i = 0; i < size; i++) {
        new_storage[i] = old_env[i];
        // Initialize ownership: startup strings are not owned by us
        new_ownership[i] = EnvStringOwnership();
      }
    }
    new_storage[size] = nullptr;

    storage = new_storage;
    ownership = new_ownership;
    capacity = new_capacity;
    is_ours = true;

    // Update app.env_ptr to point to our storage
    LIBC_NAMESPACE::app.env_ptr = reinterpret_cast<uintptr_t *>(storage);

    return true;
  }

  // We already own environ, check if we need to grow it
  if (needed <= capacity)
    return true;

  // Grow capacity by the growth factor
  size_t new_capacity = needed * ENVIRON_GROWTH_FACTOR;

  // Use realloc to grow the arrays
  char **new_storage = reinterpret_cast<char **>(
      realloc(storage, sizeof(char *) * (new_capacity + 1)));
  if (!new_storage)
    return false;

  EnvStringOwnership *new_ownership = reinterpret_cast<EnvStringOwnership *>(
      realloc(ownership, sizeof(EnvStringOwnership) * (new_capacity + 1)));
  if (!new_ownership) {
    // If ownership realloc fails, we still have the old storage in new_storage
    // which was successfully reallocated. We need to restore or handle this.
    // For safety, we'll keep the successfully reallocated storage.
    storage = new_storage;
    return false;
  }

  storage = new_storage;
  ownership = new_ownership;
  capacity = new_capacity;

  // Update app.env_ptr to point to our new storage
  LIBC_NAMESPACE::app.env_ptr = reinterpret_cast<uintptr_t *>(storage);

  return true;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
