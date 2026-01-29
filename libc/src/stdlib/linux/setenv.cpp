//===-- Implementation of setenv ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/setenv.h"
#include "../environ_internal.h"
#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

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

  // Get the singleton environment manager
  auto &env_mgr = internal::EnvironmentManager::instance();

  // Initialize environ if not already done
  env_mgr.init();

  // Search for existing variable
  int index = env_mgr.find_var(name_view);

  if (index >= 0) {
    // Variable exists
    if (overwrite == 0) {
      // Don't overwrite, just return success
      return 0;
    }

    // Need to replace the value. Ensure we own the environ storage and have
    // ownership tracking before updating in-place.
    if (!env_mgr.ensure_capacity(env_mgr.get_size())) {
      libc_errno = ENOMEM;
      return -1;
    }

    // Calculate size for "name=value" string
    size_t name_len = LIBC_NAMESPACE::internal::string_length(name);
    size_t value_len = LIBC_NAMESPACE::internal::string_length(value);
    size_t total_len =
        name_len + 1 + value_len + 1; // name + '=' + value + '\0'

    char *new_string = reinterpret_cast<char *>(malloc(total_len));
    if (!new_string) {
      libc_errno = ENOMEM;
      return -1;
    }

    // Build "name=value" string
    LIBC_NAMESPACE::inline_memcpy(new_string, name, name_len);
    new_string[name_len] = '=';
    LIBC_NAMESPACE::inline_memcpy(new_string + name_len + 1, value, value_len);
    new_string[name_len + 1 + value_len] = '\0';

    // Replace in environ array
    char **env_array = env_mgr.get_array();

    // Free old string if we allocated it
    if (env_mgr.get_ownership()[index].can_free()) {
      free(env_array[index]);
    }

    env_array[index] = new_string;
    // Mark this string as allocated by us
    env_mgr.get_ownership()[index].allocated_by_us = true;

    return 0;
  }

  // Variable doesn't exist, need to add it
  // Ensure we have capacity for one more entry
  if (!env_mgr.ensure_capacity(env_mgr.get_size() + 1)) {
    libc_errno = ENOMEM;
    return -1;
  }

  // Calculate size for "name=value" string
  size_t name_len = LIBC_NAMESPACE::internal::string_length(name);
  size_t value_len = LIBC_NAMESPACE::internal::string_length(value);
  size_t total_len = name_len + 1 + value_len + 1; // name + '=' + value + '\0'

  char *new_string = reinterpret_cast<char *>(malloc(total_len));
  if (!new_string) {
    libc_errno = ENOMEM;
    return -1;
  }

  // Build "name=value" string
  LIBC_NAMESPACE::inline_memcpy(new_string, name, name_len);
  new_string[name_len] = '=';
  LIBC_NAMESPACE::inline_memcpy(new_string + name_len + 1, value, value_len);
  new_string[name_len + 1 + value_len] = '\0';

  // Add to environ array
  char **env_array = env_mgr.get_array();
  size_t current_size = env_mgr.get_size();
  env_array[current_size] = new_string;

  // Mark this string as allocated by us
  env_mgr.get_ownership()[current_size].allocated_by_us = true;

  // Increment size and maintain null terminator
  env_mgr.increment_size();
  env_array[current_size + 1] = nullptr; // Maintain null terminator

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
