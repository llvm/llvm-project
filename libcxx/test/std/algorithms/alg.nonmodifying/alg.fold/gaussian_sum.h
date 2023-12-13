//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_FOLD_GAUSSIAN_SUM_H
#define LIBCXX_TEST_FOLD_GAUSSIAN_SUM_H

#include <vector>

template <class T>
constexpr auto gaussian_sum(std::vector<T> const& input) {
  return (static_cast<double>(input.size()) / 2) * (input.front() + input.back());
}

#endif // LIBCXX_TEST_FOLD_GAUSSIAN_SUM_H
