//===--- A data type for thread attributes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_ATTRIB_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_ATTRIB_H

#include "src/__support/architectures.h"

namespace __llvm_libc {

#if (defined(LLVM_LIBC_ARCH_AARCH64) || defined(LLVM_LIBC_ARCH_X86_64))
constexpr unsigned int STACK_ALIGNMENT = 16;
#endif
// TODO: Provide stack alignment requirements for other architectures.

// A data type to hold common thread attributes which have to be stored as
// thread state. Note that this is different from public attribute types like
// pthread_attr_t which might contain information which need not be saved as
// part of a thread's state. For example, the stack guard size.
//
// Thread attributes are typically stored on the stack. So, we align as required
// for the target architecture.
template <typename ReturnType>
struct alignas(STACK_ALIGNMENT) ThreadAttributes {
  bool detached;
  void *stack;                   // Pointer to the thread stack
  unsigned long long stack_size; // Size of the stack
  unsigned char owned_stack; // Indicates if the thread owns this stack memory
  ReturnType retval;         // The return value of thread runner is saved here
  int tid;
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_ATTRIB_H
