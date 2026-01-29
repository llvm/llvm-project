//===-- Implementation of getenv ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"
#include "environ_internal.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, getenv, (const char *name)) {
  if (name == nullptr || name[0] == '\0')
    return nullptr;

  auto &env_mgr = internal::EnvironmentManager::get_instance();
  env_mgr.init();

  cpp::string_view name_view(name);
  int idx = env_mgr.find_var(name_view);
  if (idx < 0)
    return nullptr;

  return env_mgr.get_array()[idx] + name_view.size() + 1;
}

} // namespace LIBC_NAMESPACE_DECL
