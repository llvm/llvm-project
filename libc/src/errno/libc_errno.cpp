//===-- Implementation of errno -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace __llvm_libc {

extern "C" {
// TODO: Declare __llvmlibc_errno only under LIBC_COPT_PUBLIC_PACKAGING and
// __llvmlibc_internal_errno otherwise.
//
// In overlay mode, this will be an unused thread local variable as libc_errno
// will resolve to errno from the system libc's errno.h. In full build mode
// however, libc_errno will resolve to this thread local variable via the errno
// macro defined in LLVM libc's public errno.h header file.
// TODO: Use a macro to distinguish full build and overlay build which can be
//       used to exclude __llvmlibc_errno under overlay build.
thread_local int __llvmlibc_errno;
thread_local int __llvmlibc_internal_errno;
} // extern "C"

} // namespace __llvm_libc
