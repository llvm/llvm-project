//===-- Internal utilities for environment management -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ENVIRON_INTERNAL_H
#define LLVM_LIBC_SRC_STDLIB_ENVIRON_INTERNAL_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Ownership information for environment strings.
// We need to track ownership because environment strings come from three
// sources:
// 1. Startup environment (from program loader) - we don't own these
// 2. putenv() calls where caller provides the string - we don't own these
// 3. setenv() calls where we allocate the string - we DO own these
// Only strings we allocated can be freed when replaced or removed.
struct EnvStringOwnership {
  bool allocated_by_us; // True if we malloc'd this string (must free).
                        // False for startup environ or putenv strings (don't
                        // free).

  // Default: not owned by us (startup or putenv - don't free).
  LIBC_INLINE EnvStringOwnership() : allocated_by_us(false) {}

  // Returns true if this string can be safely freed.
  LIBC_INLINE bool can_free() const { return allocated_by_us; }
};

// Centralized manager for environment variable operations.
// This class encapsulates all state and operations related to environment
// management, including memory management and tracking of string ownership.
class EnvironmentManager {
  // Our allocated environ array (nullptr if using startup environ)
  char **storage = nullptr;

  // Parallel array tracking ownership of each environ string
  // Same size/capacity as storage
  EnvStringOwnership *ownership = nullptr;

  // Allocated capacity of storage
  size_t capacity = 0;

  // Current number of variables in environ
  size_t size = 0;

  // True if we have initialized from the startup environment
  bool initialized = false;

  // True if we allocated storage (and are responsible for freeing it)
  bool is_ours = false;

  LIBC_INLINE EnvironmentManager() = default;
  LIBC_INLINE ~EnvironmentManager() = default;

public:
  // Get the singleton instance of the environment manager
  LIBC_INLINE static EnvironmentManager &get_instance() {
    static EnvironmentManager mgr;
    return mgr;
  }

  // Delete copy and move operations to enforce singleton pattern
  EnvironmentManager(const EnvironmentManager &) = delete;
  EnvironmentManager &operator=(const EnvironmentManager &) = delete;
  EnvironmentManager(EnvironmentManager &&) = delete;
  EnvironmentManager &operator=(EnvironmentManager &&) = delete;

  // Get the current size of the environment
  LIBC_INLINE size_t get_size() const { return size; }

  // Get the ownership array for direct access
  LIBC_INLINE EnvStringOwnership *get_ownership() { return ownership; }

  // Increment the size counter (used after adding a new variable)
  LIBC_INLINE void increment_size() { size++; }

  // Search for a variable by name in the current environ array.
  // Returns the index if found, or -1 if not found.
  int find_var(cpp::string_view name);

  // Ensure environ has capacity for at least `needed` entries (plus null
  // terminator). May allocate or reallocate storage. Returns true on
  // success, false on allocation failure.
  bool ensure_capacity(size_t needed);

  // Get a pointer to the current environ array.
  // This may be app.env_ptr (startup environ) or storage (our copy).
  char **get_array();

  // Initialize environ management from the startup environment.
  // This must be called before any setenv/unsetenv operations.
  void init();
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_ENVIRON_INTERNAL_H
