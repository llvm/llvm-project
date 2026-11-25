//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of syscall.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/syscall.h"

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

namespace {

// Like other libcs, syscall always reads six long-sized arguments even when
// the caller passed fewer. Reading past the end of a va_list is technically
// UB (FIXME: this could interact badly with LTO / ExpandVariadics if enabled
// on more targets), though on the ABIs we support the extra values are just
// harmless garbage ignored by the kernel past a syscall's arity. When
// arguments spill onto the stack (e.g. arg6 on x86_64), reading past the
// caller-supplied arguments may touch the caller's stack redzone under
// ASan/HWASan/TSan, so disable OOB sanitizer checks for the va_arg reads.
LIBC_NO_SANITIZE_OOB_ACCESS LIBC_INLINE long next_syscall_arg(va_list *vlist) {
  return va_arg(*vlist, long);
}

} // namespace

LLVM_LIBC_FUNCTION(long, syscall, (long number, ...)) {
  va_list vlist;
  va_start(vlist, number);

  // Callers must pass every argument as a long-sized value (pointers, long, or
  // a cast) and split 64-bit arguments themselves on 32-bit targets.
  long arg1 = next_syscall_arg(&vlist);
  long arg2 = next_syscall_arg(&vlist);
  long arg3 = next_syscall_arg(&vlist);
  long arg4 = next_syscall_arg(&vlist);
  long arg5 = next_syscall_arg(&vlist);
  long arg6 = next_syscall_arg(&vlist);
  va_end(vlist);

  ErrorOr<long> syscall_result = linux_syscalls::syscall_checked<long>(
      number, arg1, arg2, arg3, arg4, arg5, arg6);
  if (!syscall_result.has_value()) {
    libc_errno = syscall_result.error();
    return -1;
  }
  return syscall_result.value();
}

} // namespace LIBC_NAMESPACE_DECL
