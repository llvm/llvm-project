//===-- Implementation of errno -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

namespace __llvm_libc {

#ifdef LIBC_TARGET_ARCH_IS_GPU
struct ErrnoConsumer {
  void operator=(int) {}
};
#endif

extern "C" {
#ifdef LIBC_COPT_PUBLIC_PACKAGING
// TODO: Declare __llvmlibc_errno only under LIBC_COPT_PUBLIC_PACKAGING and
// __llvmlibc_internal_errno otherwise.
// In overlay mode, this will be an unused thread local variable as libc_errno
// will resolve to errno from the system libc's errno.h. In full build mode
// however, libc_errno will resolve to this thread local variable via the errno
// macro defined in LLVM libc's public errno.h header file.
// TODO: Use a macro to distinguish full build and overlay build which can be
//       used to exclude __llvmlibc_errno under overlay build.
#ifdef LIBC_TARGET_ARCH_IS_GPU
ErrnoConsumer __llvmlibc_errno;
#else
LIBC_THREAD_LOCAL int __llvmlibc_errno;
#endif // LIBC_TARGET_ARCH_IS_GPU
#else
LIBC_THREAD_LOCAL int __llvmlibc_internal_errno;
#endif
} // extern "C"

} // namespace __llvm_libc
