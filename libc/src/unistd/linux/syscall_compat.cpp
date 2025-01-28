//===-- Linux implementation of syscall (va compat) -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/syscall_compat.h"
#include "src/__support/common.h"
#include "src/unistd/syscall.h"

#include "src/__support/macros/config.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

#undef syscall
LLVM_LIBC_FUNCTION(long, syscall, (long n, ...)) {
  va_list args;
  va_start(args, n);
  long arg1 = va_arg(args, long);
  long arg2 = va_arg(args, long);
  long arg3 = va_arg(args, long);
  long arg4 = va_arg(args, long);
  long arg5 = va_arg(args, long);
  long arg6 = va_arg(args, long);
  va_end(args);
  return __llvm_libc_syscall(n, arg1, arg2, arg3, arg4, arg5, arg6);
}

} // namespace LIBC_NAMESPACE_DECL
