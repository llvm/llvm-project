//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the POSIX unsetenv function.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/unsetenv.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/stdlib/environ_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, unsetenv, (const char *name)) {
  LIBC_CRASH_ON_NULLPTR(name);

  cpp::string_view name_view(name);

  // POSIX: name must not be empty or contain '='.
  if (name_view.empty() ||
      name_view.find_first_of('=') != cpp::string_view::npos) {
    libc_errno = EINVAL;
    return -1;
  }

  int result = internal::EnvironmentManager::get_instance().unset(name_view);
  if (result != 0)
    libc_errno = ENOMEM;

  return result;
}

} // namespace LIBC_NAMESPACE_DECL
