//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of syscall
///
//===----------------------------------------------------------------------===//

#include "src/compat/syscall.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

// Makes a system call by number, for the things a libc has no function for.
// Six arguments are always read: the kernel ignores what a call does not take,
// and a caller passing fewer leaves the rest holding whatever was there, which
// is what every other libc does here too.
// A call may move the thread pointer, as Wine's signal handlers do to leave
// the 32 bit TEB, so a canary loaded through it on entry would not match the
// one checked on the way out.
#ifdef LIBC_TARGET_ARCH_IS_X86
__attribute__((no_stack_protector))
#endif
LLVM_LIBC_FUNCTION(long, syscall, (long number, ...)) {
  va_list args;
  va_start(args, number);
  long a0 = va_arg(args, long);
  long a1 = va_arg(args, long);
  long a2 = va_arg(args, long);
  long a3 = va_arg(args, long);
  long a4 = va_arg(args, long);
  long a5 = va_arg(args, long);
  va_end(args);

  long result =
      LIBC_NAMESPACE::syscall_impl<long>(number, a0, a1, a2, a3, a4, a5);
  // The kernel reports a failure as the negated error number; a libc reports
  // it in errno and answers with minus one.
  if (result < 0 && result > -4096) {
    libc_errno = static_cast<int>(-result);
    return -1;
  }
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
