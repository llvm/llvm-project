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
  auto &env_mgr = internal::EnvironmentManager::get_instance();

  // Initialize environ if not already done
  env_mgr.init();

  // Search for existing variable
  int index = env_mgr.find_var(name_view);

  if (index >= 0 && overwrite == 0) {
    return 0;
  }

  // We either need to replace an existing variable or add a new one.
  // In both cases, we must ensure we have our own environment storage
  // and enough capacity.
  size_t current_size = env_mgr.get_size();
  size_t needed_size = (index >= 0) ? current_size : current_size + 1;

  if (!env_mgr.ensure_capacity(needed_size)) {
    libc_errno = ENOMEM;
    return -1;
  }

  // Calculate size for "name=value" string
  size_t name_len = name_view.size();
  size_t value_len = LIBC_NAMESPACE::internal::string_length(value);
  size_t total_len = name_len + 1 + value_len + 1; // name + '=' + value + '\0'

  char *new_string = static_cast<char *>(malloc(total_len));
  if (!new_string) {
    libc_errno = ENOMEM;
    return -1;
  }

  // Build "name=value" string
  LIBC_NAMESPACE::inline_memcpy(new_string, name, name_len);
  new_string[name_len] = '=';
  LIBC_NAMESPACE::inline_memcpy(new_string + name_len + 1, value, value_len);
  new_string[name_len + 1 + value_len] = '\0';

  char **env_array = env_mgr.get_array();
  if (index >= 0) {
    // Replace existing variable
    if (env_mgr.get_ownership()[index].can_free())
      free(env_array[index]);

    env_array[index] = new_string;
    env_mgr.get_ownership()[index].allocated_by_us = true;
  } else {
    // Add new variable
    env_array[current_size] = new_string;
    env_mgr.get_ownership()[current_size].allocated_by_us = true;
    env_mgr.increment_size();
    env_array[current_size + 1] = nullptr; // Maintain null terminator
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
