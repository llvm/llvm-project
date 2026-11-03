//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of execlp.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/execlp.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/unistd/environ.h"
#include "src/unistd/linux/exec_utils.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, execlp, (const char *file, const char *arg0, ...)) {
  LIBC_CRASH_ON_NULLPTR(file);
  LIBC_CRASH_ON_NULLPTR(arg0);

  va_list varargs;
  va_start(varargs, arg0);
  auto ret =
      internal::exec_with_arg_list(arg0, &varargs, [&](char *const argv[]) {
        return internal::execvpe(
            file, argv, const_cast<char *const *>(LIBC_NAMESPACE::environ));
      });
  va_end(varargs);

  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }

  // Control will not reach here on success but having a return statement will
  // keep the compilers happy.
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
