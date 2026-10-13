//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of internal environment management utilities.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/environ_internal.h"
#include "config/app.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/checked_ptr.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_strcmp.h"
#ifdef LIBC_COPT_SUPPORT_ENVIRON
#include "src/unistd/environ.h"
#endif

#if defined(LIBC_COPT_ENVIRON_LOCKING) &&                                      \
    (LIBC_THREAD_MODE == LIBC_THREAD_MODE_PLATFORM)
#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/mutex.h"
#define LIBC_ENVIRON_USE_LOCKING 1
#else
#define LIBC_ENVIRON_USE_LOCKING 0
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {

namespace {

// Synchronization policy for EnvironmentManager.  When platform mutex support
// is available, uses a recursive mutex (to avoid self-deadlock when setenv()
// triggers allocator calls that re-enter getenv()) and atomic double-checked
// initialization.  Otherwise, locking and initialization checks degrade to
// zero-overhead no-ops so environment APIs remain available without threads.
template <bool Enabled> struct EnvSync;

template <> struct EnvSync<false> {
  struct NoOpMutex {
    LIBC_INLINE constexpr NoOpMutex() = default;
    LIBC_INLINE void lock() {}
    LIBC_INLINE void unlock() {}
  };

  NoOpMutex lock{};
  bool initialized = false;

  LIBC_INLINE constexpr EnvSync() = default;
  LIBC_INLINE bool is_initialized() { return initialized; }
  LIBC_INLINE bool is_initialized_relaxed() { return initialized; }
  LIBC_INLINE void mark_initialized() { initialized = true; }
};

#if LIBC_ENVIRON_USE_LOCKING
template <> struct EnvSync<true> {
  Mutex lock;
  cpp::Atomic<bool> initialized{false};

  LIBC_INLINE constexpr EnvSync()
      : lock(/*is_priority_inherit=*/false, /*is_recursive=*/true,
             /*is_robust=*/false, /*is_pshared=*/false),
        initialized(false) {}
  LIBC_INLINE bool is_initialized() {
    return initialized.load(cpp::MemoryOrder::ACQUIRE);
  }
  LIBC_INLINE bool is_initialized_relaxed() {
    return initialized.load(cpp::MemoryOrder::RELAXED);
  }
  LIBC_INLINE void mark_initialized() {
    initialized.store(true, cpp::MemoryOrder::RELEASE);
  }
};
#endif

LIBC_CONSTINIT EnvSync<static_cast<bool>(LIBC_ENVIRON_USE_LOCKING)> env_sync;

} // namespace

// Minimum initial capacity for the environment array when first allocated.
// This avoids frequent reallocations for small environments.
constexpr size_t MIN_ENVIRON_CAPACITY = 32;

// Growth factor for environment array capacity when expanding.
// When capacity is exceeded, new_capacity = old_capacity *
// ENVIRON_GROWTH_FACTOR.
constexpr size_t ENVIRON_GROWTH_FACTOR = 2;

void EnvironmentManager::init_once() {
  if (env_sync.is_initialized())
    return;

  cpp::lock_guard lock(env_sync.lock);
  if (env_sync.is_initialized_relaxed())
    return;

  // Count entries in the startup environ.
  char **env_ptr = reinterpret_cast<char **>(app.env_ptr);
  if (env_ptr) {
    size_t c = 0;
    for (char **env = env_ptr; *env != nullptr; env++)
      c++;
    count = c;
  }

  env_sync.mark_initialized();
}

EnvironmentManager &EnvironmentManager::get_instance() {
  static LIBC_CONSTINIT EnvironmentManager mgr;
  mgr.init_once();
  return mgr;
}

char **EnvironmentManager::get_array() {
  if (is_ours)
    return storage;
  return reinterpret_cast<char **>(app.env_ptr);
}

EnvironmentManager::iterator EnvironmentManager::begin() {
  cpp::lock_guard lock(env_sync.lock);
  return get_array();
}

EnvironmentManager::iterator EnvironmentManager::end() {
  cpp::lock_guard lock(env_sync.lock);
  char **env_array = get_array();
  return env_array ? env_array + count : nullptr;
}

size_t EnvironmentManager::size() {
  cpp::lock_guard lock(env_sync.lock);
  return count;
}

char *EnvironmentManager::get(cpp::string_view name) {
  cpp::lock_guard lock(env_sync.lock);

  cpp::optional<size_t> idx = find_var(name);
  if (!idx)
    return nullptr;
  return get_array()[*idx] + name.size() + 1;
}

cpp::optional<size_t> EnvironmentManager::find_var(cpp::string_view name) {
  char **env_array = get_array();
  if (!env_array)
    return cpp::nullopt;

  auto comp = [](char l, char r) -> int {
    return static_cast<unsigned char>(l) - static_cast<unsigned char>(r);
  };

  // Empty names, supported by putenv, match entries starting with '='.
  const char first = name.empty() ? '=' : name.front();
  if (first == '\0')
    return cpp::nullopt;
  const size_t start = name.empty() ? 0 : 1;

  for (size_t i = 0; i < count; ++i) {
    const char *entry = env_array[i];
    if (entry[0] != first)
      continue;

    if (inline_strncmp(entry + start, name.data() + start, name.size() - start,
                       comp) == 0 &&
        entry[name.size()] == '=')
      return i;
  }

  return cpp::nullopt;
}

bool EnvironmentManager::ensure_capacity(size_t needed) {
  // Nothing to do if we already own an array that is large enough.
  if (is_ours && needed <= capacity)
    return true;

  // Either transition from the startup environ (pointed to by app.env_ptr) to
  // our own managed storage, which allows us to track ownership of strings and
  // safely expand the array, or grow the array we already own. We avoid
  // realloc to ensure that failures don't leave the manager in an inconsistent
  // state.
  size_t new_capacity = (!is_ours && needed < MIN_ENVIRON_CAPACITY)
                            ? MIN_ENVIRON_CAPACITY
                            : needed * ENVIRON_GROWTH_FACTOR;

  auto new_storage = make_unique_for_overwrite<char *[]>(new_capacity + 1);
  if (!new_storage)
    return false;
  // All ownership entries start out as not-owned.
  auto new_ownership = make_unique<EnvStringOwnership[]>(new_capacity + 1);
  if (!new_ownership)
    return false;

  char **old_storage = get_array();
  for (size_t i = 0; i < count; i++) {
    new_storage[i] = old_storage[i];
    if (is_ours)
      new_ownership[i] = ownership[i];
  }
  new_storage[count] = nullptr;

  if (is_ours) {
    // Old storage is intentionally retained so callers holding a reference
    // to the previous environ array do not encounter use-after-free.
    // We free ownership since it is internal to EnvironmentManager.
    delete[] ownership;
    LIBC_LSAN_IGNORE_OBJECT(storage);
  }
  storage = new_storage.release();
  ownership = new_ownership.release();
  capacity = new_capacity;
  is_ours = true;

  // Update the global environ pointer.
  app.env_ptr = reinterpret_cast<uintptr_t *>(storage);
#ifdef LIBC_COPT_SUPPORT_ENVIRON
  environ = storage;
#endif

  return true;
}

int EnvironmentManager::set(cpp::string_view name, cpp::string_view value,
                            bool overwrite) {
  cpp::lock_guard lock(env_sync.lock);

  cpp::optional<size_t> idx = find_var(name);

  // If the variable exists and we're not overwriting, do nothing.
  if (idx && !overwrite)
    return 0;

  // Ensure we have capacity. If the variable doesn't exist, we need one
  // more slot.
  size_t needed = idx ? count : count + 1;
  if (!ensure_capacity(needed))
    return -1;

  // Build the "name=value" string.
  size_t name_len = name.size();
  size_t value_len = value.size();
  size_t total_len = name_len + 1 + value_len + 1; // name + '=' + value + '\0'

  auto new_string = make_unique_for_overwrite<char[]>(total_len);
  if (!new_string)
    return -1;

  inline_memcpy(new_string.get(), name.data(), name_len);
  new_string[name_len] = '=';
  inline_memcpy(new_string.get() + name_len + 1, value.data(), value_len);
  new_string[name_len + 1 + value_len] = '\0';

  char **env_array = get_array();

  if (idx) {
    // Replace existing variable. Free old string if we own it.
    if (ownership[*idx].can_free())
      delete[] env_array[*idx];

    env_array[*idx] = new_string.release();
    ownership[*idx].allocated_by_us = true;
  } else {
    // Add new variable at the end.
    env_array[count] = new_string.release();
    ownership[count].allocated_by_us = true;
    count++;
    env_array[count] = nullptr; // Maintain null terminator.
  }

  return 0;
}

int EnvironmentManager::unset(cpp::string_view name) {
  cpp::lock_guard lock(env_sync.lock);
  return unset_unlocked(name);
}

int EnvironmentManager::unset_unlocked(cpp::string_view name) {
  cpp::optional<size_t> idx = find_var(name);
  if (!idx)
    return 0; // Variable not found; POSIX defines this as success.

  // Transition to managed storage so we can modify the array and track
  // ownership correctly.
  if (!ensure_capacity(count))
    return -1;

  char **env_array = get_array();

  // Loop to remove all instances of the variable (e.g. duplicates from execve).
  while (idx) {
    size_t i = *idx;
    if (ownership[i].can_free())
      delete[] env_array[i];

    // Compact: shift remaining entries left to fill the gap.
    // Shifting elements preserves the order of environment variables, which is
    // desirable to maintain consistency with getenv resolution order and
    // typical environ iteration behavior.
    for (size_t j = i; j < count - 1; j++) {
      env_array[j] = env_array[j + 1];
      ownership[j] = ownership[j + 1];
    }
    count--;
    env_array[count] = nullptr;

    idx = find_var(name);
  }

  return 0;
}

int EnvironmentManager::put(char *string) {
  cpp::lock_guard lock(env_sync.lock);

  cpp::string_view sv(string);
  size_t eq_pos = sv.find_first_of('=');

  // No '=' found: treat as unset (POSIX behavior).
  if (eq_pos == cpp::string_view::npos)
    return unset_unlocked(sv);

  cpp::string_view name = sv.substr(0, eq_pos);

  cpp::optional<size_t> idx = find_var(name);

  size_t needed = idx ? count : count + 1;
  if (!ensure_capacity(needed))
    return -1;

  char **env_array = get_array();

  if (idx) {
    // Replace existing variable. Free old string if we own it.
    if (ownership[*idx].can_free())
      delete[] env_array[*idx];

    env_array[*idx] = string;
    ownership[*idx].allocated_by_us = false; // Caller owns this string.
  } else {
    // Add new variable at the end.
    env_array[count] = string;
    ownership[count].allocated_by_us = false;
    count++;
    env_array[count] = nullptr;
  }

  return 0;
}

int EnvironmentManager::clear() {
  cpp::lock_guard lock(env_sync.lock);

  // Free the pointer array, but leave the string buffers intact so that
  // existing getenv pointers remain valid.
  if (storage) {
    for (char *entry : cpp::span<char *>(storage, count))
      LIBC_LSAN_IGNORE_OBJECT(entry);
  }
  delete[] storage;
  delete[] ownership;
  storage = nullptr;
  ownership = nullptr;
  capacity = 0;
  count = 0;
  is_ours = false;

  app.env_ptr = nullptr;
#ifdef LIBC_COPT_SUPPORT_ENVIRON
  environ = nullptr;
#endif

  return 0;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
