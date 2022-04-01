//===--- A data type for thread attributes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_ATTRIB_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_ATTRIB_H

namespace __llvm_libc {

// A data type to hold common thread attributes which have to be stored as
// thread state. A platform thread implementation should store the attrib object
// in its Thread data structure. Note that this is different from public
// attribute types like pthread_attr which contain information which need not
// be saved as part of a thread's state. For example, the stack guard size.
template <typename ReturnType> struct ThreadAttributes {
  void *stack;                   // Pointer to the thread stack
  unsigned long long stack_size; // Size of the stack
  unsigned char owned_stack; // Indicates if the thread owns this stack memory
  ReturnType retval;         // The return value of thread runner is saved here
  int tid;
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_ATTRIB_H
