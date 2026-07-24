//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the ptrace function.
///
//===----------------------------------------------------------------------===//

#include "src/sys/ptrace/ptrace.h"

#include "hdr/sys_ptrace_macros.h"
#include "hdr/types/pid_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ptrace.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, ptrace, (int request, ...)) {
  // Unlike the C interface, the kernel returns the peek result through the
  // fourth argument. This means the user needs to clear errno to distinguish
  // failure from a legitimate "-1" return.
  bool is_peek = request == PTRACE_PEEKTEXT || request == PTRACE_PEEKDATA ||
                 request == PTRACE_PEEKUSER;
  long peek_result;

  va_list args;
  va_start(args, request);
  pid_t pid = va_arg(args, pid_t);
  void *addr = va_arg(args, void *);
  void *data = is_peek ? &peek_result : va_arg(args, void *);
  va_end(args);

  ErrorOr<long> result = linux_syscalls::ptrace(request, pid, addr, data);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return is_peek ? peek_result : result.value();
}

} // namespace LIBC_NAMESPACE_DECL
