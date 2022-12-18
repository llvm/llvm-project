//===-- Libc specific custom operator new and delete ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_NEW_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_NEW_H

#include <stddef.h> // For size_t
#include <stdlib.h> // For malloc, free etc.

// Defining members in the std namespace is not preferred. But, we do it here
// so that we can use it to define the operator new which takes std::align_val_t
// argument.
namespace std {

enum class align_val_t : size_t {};

} // namespace std

namespace __llvm_libc {

class AllocChecker {
  bool success = false;
  AllocChecker &operator=(bool status) {
    success = status;
    return *this;
  }

public:
  AllocChecker() = default;
  operator bool() const { return success; }

  static void *alloc(size_t s, AllocChecker &ac) {
    void *mem = ::malloc(s);
    ac = (mem != nullptr);
    return mem;
  }

  static void *aligned_alloc(size_t s, std::align_val_t align,
                             AllocChecker &ac) {
    void *mem = ::aligned_alloc(static_cast<size_t>(align), s);
    ac = (mem != nullptr);
    return mem;
  }
};

} // namespace __llvm_libc

inline void *operator new(size_t size, __llvm_libc::AllocChecker &ac) noexcept {
  return __llvm_libc::AllocChecker::alloc(size, ac);
}

inline void *operator new(size_t size, std::align_val_t align,
                          __llvm_libc::AllocChecker &ac) noexcept {
  return __llvm_libc::AllocChecker::aligned_alloc(size, align, ac);
}

inline void *operator new[](size_t size,
                            __llvm_libc::AllocChecker &ac) noexcept {
  return __llvm_libc::AllocChecker::alloc(size, ac);
}

inline void *operator new[](size_t size, std::align_val_t align,
                            __llvm_libc::AllocChecker &ac) noexcept {
  return __llvm_libc::AllocChecker::aligned_alloc(size, align, ac);
}

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_NEW_H
