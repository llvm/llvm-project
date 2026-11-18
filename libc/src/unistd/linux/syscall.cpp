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
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, syscall, (long number, ...)) {
  va_list vlist;
  va_start(vlist, number);
  internal::ArgList args(vlist);
  va_end(vlist);

  // Like other libcs, this always reads six long-sized arguments, even when
  // the caller passed fewer. Reading past the end of a va_list is technically
  // UB (FIXME: this could interact badly with LTO / ExpandVariadics if enabled
  // on more targets), though on the ABIs we support the extra values are just
  // harmless garbage ignored by the kernel past a syscall's arity. Callers must
  // pass every argument as a long-sized value (pointers, long, or a cast) and
  // split 64-bit arguments themselves on 32-bit targets.
  long arg1 = args.next_var<long>();
  long arg2 = args.next_var<long>();
  long arg3 = args.next_var<long>();
  long arg4 = args.next_var<long>();
  long arg5 = args.next_var<long>();
  long arg6 = args.next_var<long>();

  ErrorOr<long> syscall_result = linux_syscalls::syscall_checked<long>(
      number, arg1, arg2, arg3, arg4, arg5, arg6);
  if (!syscall_result.has_value()) {
    libc_errno = syscall_result.error();
    return -1;
  }
  return syscall_result.value();
}

} // namespace LIBC_NAMESPACE_DECL
