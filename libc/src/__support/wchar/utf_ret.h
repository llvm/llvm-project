//===-- Definition of utf_ret ----------------------------------*-- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_UTF_RET_H
#define LLVM_LIBC_SRC___SUPPORT_UTF_RET_H

namespace LIBC_NAMESPACE_DECL {

template <typename T> struct utf_ret {
  T out;
  int error;
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_UTF_RET_H
