//===-- String to float conversion utils ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_STR_TO_FLOAT_H
#define LLVM_LIBC_SHARED_STR_TO_FLOAT_H

#include "src/__support/str_to_float.h"

namespace LIBC_NAMESPACE::shared {

// WARNING: This is a proof of concept. In future the interface point for libcxx
// won't be using libc internal classes.

template <class T>
inline internal::FloatConvertReturn<T> decimal_exp_to_float(
    internal::ExpandedFloat<T> init_num, bool truncated,
    internal::RoundDirection round, const char *__restrict num_start,
    const size_t num_len = cpp::numeric_limits<size_t>::max()) {
  return internal::decimal_exp_to_float(init_num, truncated, round, num_start,
                                        num_len);
}
} // namespace LIBC_NAMESPACE::shared

#endif // LLVM_LIBC_SHARED_STR_TO_FLOAT_H
