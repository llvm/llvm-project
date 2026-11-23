//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the POSIX getenv function.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/environ_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, getenv, (const char *name)) {
  if (name == nullptr || name[0] == '\0')
    return nullptr;

  return internal::EnvironmentManager::get_instance().get(name);
}

} // namespace LIBC_NAMESPACE_DECL
