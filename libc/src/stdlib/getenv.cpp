//===-- Implementation of getenv ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"
#include "config/linux/app.h"
#include "src/__support/CPP/StringView.h"
#include "src/__support/common.h"

#include <stddef.h> // For size_t.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, getenv, (const char *name)) {
  char **env_ptr = reinterpret_cast<char **>(__llvm_libc::app.envPtr);

  if (name == nullptr || env_ptr == nullptr)
    return nullptr;

  __llvm_libc::cpp::StringView env_var_name(name);
  if (env_var_name.size() == 0)
    return nullptr;
  for (char **env = env_ptr; *env != nullptr; env++) {
    __llvm_libc::cpp::StringView cur(*env);
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

} // namespace __llvm_libc
