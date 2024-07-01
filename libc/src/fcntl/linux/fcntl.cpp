//===-- Implementation of fcntl -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/fcntl.h"

#include "src/__support/OSUtil/fcntl.h"
#include "src/__support/common.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, fcntl, (int fd, int cmd, ...)) {
  void *arg;
  va_list varargs;
  va_start(varargs, cmd);
  arg = va_arg(varargs, void *);
  va_end(varargs);
  return LIBC_NAMESPACE::internal::fcntl(fd, cmd, arg);
}

} // namespace LIBC_NAMESPACE
