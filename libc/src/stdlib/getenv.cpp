//===-- Implementation of getenv ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"
#include "config/linux/app.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"

#include <stddef.h> // For size_t.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(char *, getenv, (const char *name)) {
  char **env_ptr = reinterpret_cast<char **>(LIBC_NAMESPACE::app.env_ptr);

  if (name == nullptr || env_ptr == nullptr)
    return nullptr;

  LIBC_NAMESPACE::cpp::string_view env_var_name(name);
  if (env_var_name.size() == 0)
    return nullptr;
  for (char **env = env_ptr; *env != nullptr; env++) {
    LIBC_NAMESPACE::cpp::string_view cur(*env);
    if (!cur.starts_with(env_var_name))
      continue;

    if (cur[env_var_name.size()] != '=')
      continue;

    // Remove the name and the equals sign.
    cur.remove_prefix(env_var_name.size() + 1);
    // We know that data is null terminated, so this is safe.
    return const_cast<char *>(cur.data());
  }

  return nullptr;
}

} // namespace LIBC_NAMESPACE
