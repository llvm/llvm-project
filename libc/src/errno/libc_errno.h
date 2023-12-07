//===-- Implementation header for errno -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
#define LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

#include <errno.h>

// If we are targeting the GPU we currently don't support 'errno'. We simply
// consume it.
#ifdef LIBC_TARGET_ARCH_IS_GPU
namespace __llvm_libc {
struct ErrnoConsumer {
  void operator=(int) {}
};
} // namespace __llvm_libc
#endif

// All of the libc runtime and test code should use the "libc_errno" macro. They
// should not refer to the "errno" macro directly.
#ifdef LIBC_COPT_PUBLIC_PACKAGING
#ifdef LIBC_TARGET_ARCH_IS_GPU
extern "C" __llvm_libc::ErrnoConsumer __llvmlibc_errno;
#define libc_errno __llvmlibc_errno
#else
// This macro will resolve to errno from the errno.h file included above. Under
// full build, this will be LLVM libc's errno. In overlay build, it will be
// system libc's errno.
#define libc_errno errno
#endif
#else
namespace __llvm_libc {

// TODO: On the GPU build this will be mapped to a single global value. We need
// to ensure that tests are not run with multiple threads that depend on errno
// until we have true 'thread_local' support on the GPU.
extern "C" LIBC_THREAD_LOCAL int __llvmlibc_internal_errno;

// TODO: After all of libc/src and libc/test are switched over to use
// libc_errno, this header file will be "shipped" via an add_entrypoint_object
// target. At which point libc_errno, should point to __llvmlibc_internal_errno
// if LIBC_COPT_PUBLIC_PACKAGING is not defined.
#define libc_errno __llvm_libc::__llvmlibc_internal_errno

} // namespace __llvm_libc
#endif

#endif // LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
