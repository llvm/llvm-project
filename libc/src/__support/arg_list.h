//===-- Holder Class for manipulating va_lists ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_ARG_LIST_H
#define LLVM_LIBC_SRC_SUPPORT_ARG_LIST_H

#include <stdarg.h>

namespace __llvm_libc {
namespace internal {

class ArgList {
  va_list vlist;

public:
  ArgList(va_list vlist) { va_copy(this->vlist, vlist); }
  ArgList(ArgList &other) { va_copy(this->vlist, other.vlist); }
  ~ArgList() { va_end(this->vlist); }

  template <class T> T inline next_var() { return va_arg(vlist, T); }
};

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_ARG_LIST_H
