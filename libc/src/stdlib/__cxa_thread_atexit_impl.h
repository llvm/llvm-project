//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___THREADS_CXA_THREAD_ATEXIT_IMPL_H
#define LLVM_LIBC_SRC___THREADS_CXA_THREAD_ATEXIT_IMPL_H

#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

int __cxa_thread_atexit_impl(AtExitCallback* callback, void* obj, void*);
}  // namespace LIBC_NAMESPACE_DECL

#endif  // LLVM_LIBC_SRC_THREADS_CXA_THREAD_ATEXIT_IMPL_H
