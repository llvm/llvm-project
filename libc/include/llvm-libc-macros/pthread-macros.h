//===-- Definition of pthread macros --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_PTHREAD_MACRO_H
#define LLVM_LIBC_MACROS_PTHREAD_MACRO_H

#define PTHREAD_CREATE_JOINABLE 0
#define PTHREAD_CREATE_DETACHED 1

#define PTHREAD_MUTEX_NORMAL 0
#define PTHREAD_MUTEX_ERRORCHECK 1
#define PTHREAD_MUTEX_RECURSIVE 2
#define PTHREAD_MUTEX_DEFAULT PTHREAD_MUTEX_NORMAL

#define PTHREAD_MUTEX_STALLED 0
#define PTHREAD_MUTEX_ROBUST 1

#define PTHREAD_ONCE_INIT {0}

#define PTHREAD_PROCESS_PRIVATE 0
#define PTHREAD_PROCESS_SHARED 1

#define PTHREAD_MUTEX_INITIALIZER {0}
#define PTHREAD_RWLOCK_INITIALIZER {0}

// glibc extensions
#define PTHREAD_STACK_MIN (1 << 14) // 16KB
#define PTHREAD_RWLOCK_PREFER_READER_NP 0
#define PTHREAD_RWLOCK_PREFER_WRITER_NP 1
#define PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP 2

#endif // LLVM_LIBC_MACROS_PTHREAD_MACRO_H
