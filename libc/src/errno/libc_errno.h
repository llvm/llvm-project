//===-- Implementation header for errno -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
#define LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H

#include <errno.h>

// All of the libc runtime and test code should use the "libc_errno" macro. They
// should not refer to the "errno" macro directly.
#ifdef LIBC_COPT_PUBLIC_PACKAGING
// This macro will resolve to errno from the errno.h file included above. Under
// full build, this will be LLVM libc's errno. In overlay build, it will be
// system libc's errno.
#define libc_errno errno
#else
namespace __llvm_libc {

extern "C" {
extern thread_local int __llvmlibc_internal_errno;
} // extern "C"

// TODO: After all of libc/src and libc/test are switched over to use
// libc_errno, this header file will be "shipped" via an add_entrypoint_object
// target. At which point libc_errno, should point to __llvmlibc_internal_errno
// if LIBC_COPT_PUBLIC_PACKAGING is not defined.
#define libc_errno errno

} // namespace __llvm_libc
#endif

#endif // LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
