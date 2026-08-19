//===-- Linux implementation of execle ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/execle.h"
#include "src/__support/macros/config.h"

#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include <stdarg.h>
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, execle, (const char *path, const char *arg0, ...)) {
  va_list varargs;
  va_start(varargs, arg0);

  size_t argc = 1;
  while (va_arg(varargs, const char *) != nullptr)
    ++argc;
  va_end(varargs);

  char **argv =
      static_cast<char **>(__builtin_alloca((argc + 1) * sizeof(char *)));
  argv[0] = const_cast<char *>(arg0);

  va_start(varargs, arg0);
  for (size_t i = 1; i <= argc; ++i)
    argv[i] = va_arg(varargs, char *);
  char **envp = va_arg(varargs, char **);
  va_end(varargs);

  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_execve, path, argv, envp);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  // Control will not reach here on success but have a return statement will
  // keep the compilers happy.
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
