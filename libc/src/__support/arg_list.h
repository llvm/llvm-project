//===-- Holder Class for manipulating va_lists ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ARG_LIST_H
#define LLVM_LIBC_SRC___SUPPORT_ARG_LIST_H

#include "src/__support/common.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace internal {

class ArgList {
  va_list vlist;

public:
  LIBC_INLINE ArgList(va_list vlist) { va_copy(this->vlist, vlist); }
  LIBC_INLINE ArgList(ArgList &other) { va_copy(this->vlist, other.vlist); }
  LIBC_INLINE ~ArgList() { va_end(this->vlist); }

  LIBC_INLINE ArgList &operator=(ArgList &rhs) {
    va_copy(vlist, rhs.vlist);
    return *this;
  }

  template <class T> LIBC_INLINE T next_var() { return va_arg(vlist, T); }
};

// Used for testing things that use an ArgList when it's impossible to know what
// the arguments should be ahead of time. An example of this would be fuzzing,
// since a function passed a random input could request unpredictable arguments.
class MockArgList {
  size_t arg_counter = 0;

public:
  LIBC_INLINE MockArgList() = default;
  LIBC_INLINE MockArgList(va_list) { ; }
  LIBC_INLINE MockArgList(MockArgList &other) {
    arg_counter = other.arg_counter;
  }
  LIBC_INLINE ~MockArgList() = default;

  LIBC_INLINE MockArgList &operator=(MockArgList &rhs) {
    arg_counter = rhs.arg_counter;
    return *this;
  }

  template <class T> LIBC_INLINE T next_var() {
    ++arg_counter;
    return T(arg_counter);
  }

  size_t read_count() const { return arg_counter; }
};

// Used for the GPU implementation of `printf`. This models a variadic list as a
// simple array of pointers that are built manually by the implementation.
class StructArgList {
  void *ptr;
  void *end;

public:
  LIBC_INLINE StructArgList(void *ptr, size_t size)
      : ptr(ptr), end(reinterpret_cast<unsigned char *>(ptr) + size) {}
  LIBC_INLINE StructArgList(const StructArgList &other) {
    ptr = other.ptr;
    end = other.end;
  }
  LIBC_INLINE StructArgList() = default;
  LIBC_INLINE ~StructArgList() = default;

  LIBC_INLINE StructArgList &operator=(const StructArgList &rhs) {
    ptr = rhs.ptr;
    return *this;
  }

  LIBC_INLINE void *get_ptr() const { return ptr; }

  template <class T> LIBC_INLINE T next_var() {
    ptr = reinterpret_cast<void *>(
        ((reinterpret_cast<uintptr_t>(ptr) + alignof(T) - 1) / alignof(T)) *
        alignof(T));

    if (ptr >= end)
      return T(-1);

    T val = *reinterpret_cast<T *>(ptr);
    ptr = reinterpret_cast<unsigned char *>(ptr) + sizeof(T);
    return val;
  }
};

} // namespace internal
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_ARG_LIST_H
