//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal utilities for environment variable management.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ENVIRON_INTERNAL_H
#define LLVM_LIBC_SRC_STDLIB_ENVIRON_INTERNAL_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/optional.h"
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
  bool allocated_by_us; // True if we allocated this string (must delete).
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
//
// The manager provides iterator support, allowing callers to iterate over
// the current environment entries using standard begin()/end() semantics.
class EnvironmentManager {
  // Our allocated environ array (nullptr if using startup environ)
  char **storage = nullptr;

  // Parallel array tracking ownership of each environ string.
  // Allocated with the same capacity as storage.
  EnvStringOwnership *ownership = nullptr;

  // Allocated capacity of storage
  size_t capacity = 0;

  // Current number of variables in environ
  size_t count = 0;

  // True if we have initialized from the startup environment
  bool initialized = false;

  // True if we allocated storage (and are responsible for freeing it)
  bool is_ours = false;

  EnvironmentManager() = default;
  ~EnvironmentManager() = default;

  // Lazily initialize from the startup environment.
  // Called internally by get_instance(); idempotent.
  void init_once();

  // Get a pointer to the current environ array.
  // This may be app.env_ptr (startup environ) or storage (our copy).
  char **get_array();

  // Search for a variable by name in the current environ array.
  // Returns the index if found, or nullopt if not found.
  cpp::optional<size_t> find_var(cpp::string_view name);

  // Ensure environ has capacity for at least `needed` entries (plus null
  // terminator). May allocate or reallocate storage. Returns true on
  // success, false on allocation failure.
  bool ensure_capacity(size_t needed);

  // Helper: allocate new storage and ownership arrays of the given capacity,
  // copy the first `copy_count` entries from old_storage/old_ownership, and
  // initialize the remaining ownership slots to default (not-owned).
  // Returns false on allocation failure.
  bool alloc_and_copy(size_t new_capacity, char **old_storage,
                      EnvStringOwnership *old_ownership, size_t copy_count,
                      char **&out_storage, EnvStringOwnership *&out_ownership);

public:
  // Get the singleton instance of the environment manager.
  static EnvironmentManager &get_instance();

  // Delete copy and move operations to enforce singleton pattern.
  EnvironmentManager(const EnvironmentManager &) = delete;
  EnvironmentManager &operator=(const EnvironmentManager &) = delete;
  EnvironmentManager(EnvironmentManager &&) = delete;
  EnvironmentManager &operator=(EnvironmentManager &&) = delete;

  // Iterator support for traversing environment entries.
  using iterator = char **;
  iterator begin();
  iterator end();
  size_t size() const;

  // Look up a variable by name. Returns a pointer to the value string
  // (after the '='), or nullptr if not found.
  char *get(cpp::string_view name);
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_ENVIRON_INTERNAL_H
