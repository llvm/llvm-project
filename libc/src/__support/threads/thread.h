//===--- A platform independent indirection for a thread class --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_H

#include <stddef.h>

// The platform specific implemnetations are pulled via the following include.
// The idea is for the platform implementation to implement a class named Thread
// in the namespace __llvm_libc with the following properties:
//
// 1. Has a defaulted default constructor (not a default constructor).
//
// 2. Has a "run" method with the following signature:
//
//        int run(ThreadRunner *f, void *arg, void *stack, size_t size);
//
//    Returns:
//        0 on success and an error value on failure.
//    Args:
//        arg - The argument to be passed to the thread runner after the thread
//              is created.
//        stack - The stack to use for the thread.
//        size - The stack size.
//
//    If callers pass a non-null |stack| value, then it will assumed that
//      1. The clean up the stack memory is their responsibility
//      2. The guard area is setup appropriately by the caller.
//
// 3. Has a "join" method with the following signature:
//      ErrorOr<ReturnType> join();
//    The "join" method should return 0 on success and set retcode to the
//    threads return value. On failure, an appropriate errno value should be
//    returned.
//
// 4. Has an operator== for comparison between two threads.
#ifdef __unix__
#include "linux/thread.h"
#endif // __unix__

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_H
