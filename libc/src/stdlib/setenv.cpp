//===-- Implementation of setenv ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/setenv.h"
#include "environ_internal.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/memcpy.h"
#include "src/string/strlen.h"

// We use extern "C" declarations for malloc/free instead of including
// src/stdlib/malloc.h and src/stdlib/free.h. This allows the implementation
// to work with different allocator implementations, particularly in integration
// tests which provide a simple bump allocator. The extern "C" linkage ensures
// we use whatever allocator is linked with the test or application.
extern "C" void *malloc(size_t);
extern "C" void free(void *);

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setenv,
                   (const char *name, const char *value, int overwrite)) {
  // Validate inputs
  if (name == nullptr || value == nullptr) {
    libc_errno = EINVAL;
    return -1;
  }

  cpp::string_view name_view(name);
  if (name_view.empty()) {
    libc_errno = EINVAL;
    return -1;
  }

  // POSIX: name cannot contain '='
  if (name_view.find_first_of('=') != cpp::string_view::npos) {
    libc_errno = EINVAL;
    return -1;
  }

  // Lock mutex for thread safety
  internal::environ_mutex.lock();

  // Initialize environ if not already done
  internal::init_environ();

  // Search for existing variable
  int index = internal::find_env_var(name_view);

  if (index >= 0) {
    // Variable exists
    if (overwrite == 0) {
      // Don't overwrite, just return success
      internal::environ_mutex.unlock();
      return 0;
    }

    // Need to replace the value
    // Calculate size for "name=value" string
    size_t name_len = LIBC_NAMESPACE::strlen(name);
    size_t value_len = LIBC_NAMESPACE::strlen(value);
    size_t total_len =
        name_len + 1 + value_len + 1; // name + '=' + value + '\0'

    char *new_string = reinterpret_cast<char *>(malloc(total_len));
    if (!new_string) {
      internal::environ_mutex.unlock();
      libc_errno = ENOMEM;
      return -1;
    }

    // Build "name=value" string
    LIBC_NAMESPACE::memcpy(new_string, name, name_len);
    new_string[name_len] = '=';
    LIBC_NAMESPACE::memcpy(new_string + name_len + 1, value, value_len);
    new_string[name_len + 1 + value_len] = '\0';

    // Replace in environ array
    char **env_array = internal::get_environ_array();

    // Free old string if we allocated it
    if (internal::environ_ownership[index].can_free()) {
      free(env_array[index]);
    }

    env_array[index] = new_string;
    // Mark this string as allocated by us
    internal::environ_ownership[index].allocated_by_us = true;

    internal::environ_mutex.unlock();
    return 0;
  }

  // Variable doesn't exist, need to add it
  // Ensure we have capacity for one more entry
  if (!internal::ensure_capacity(internal::environ_size + 1)) {
    internal::environ_mutex.unlock();
    libc_errno = ENOMEM;
    return -1;
  }

  // Calculate size for "name=value" string
  size_t name_len = LIBC_NAMESPACE::strlen(name);
  size_t value_len = LIBC_NAMESPACE::strlen(value);
  size_t total_len = name_len + 1 + value_len + 1; // name + '=' + value + '\0'

  char *new_string = reinterpret_cast<char *>(malloc(total_len));
  if (!new_string) {
    internal::environ_mutex.unlock();
    libc_errno = ENOMEM;
    return -1;
  }

  // Build "name=value" string
  LIBC_NAMESPACE::memcpy(new_string, name, name_len);
  new_string[name_len] = '=';
  LIBC_NAMESPACE::memcpy(new_string + name_len + 1, value, value_len);
  new_string[name_len + 1 + value_len] = '\0';

  // Add to environ array
  char **env_array = internal::get_environ_array();
  env_array[internal::environ_size] = new_string;

  // Mark this string as allocated by us
  internal::environ_ownership[internal::environ_size].allocated_by_us = true;

  internal::environ_size++;
  env_array[internal::environ_size] = nullptr; // Maintain null terminator

  internal::environ_mutex.unlock();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
