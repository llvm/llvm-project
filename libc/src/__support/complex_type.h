//===-- complex type --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_COMPLEX_TYPE_H
#define LLVM_LIBC_SRC___SUPPORT_COMPLEX_TYPE_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
template <typename T> struct Complex {
  T real;
  T imag;
};

template <typename T, typename U>
T conjugate(T c) {
  Complex<U> c_c = cpp::bit_cast<Complex<U>>(c);
  c_c.imag = -c_c.imag;
  return cpp::bit_cast<T>(c_c);
}

} // namespace LIBC_NAMESPACE_DECL
#endif // LLVM_LIBC_SRC___SUPPORT_COMPLEX_TYPE_H
