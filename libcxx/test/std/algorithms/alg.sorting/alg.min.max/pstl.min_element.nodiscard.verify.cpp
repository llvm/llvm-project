//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <algorithm>

// template<class ExecutionPolicy, class ForwardIterator>
//   [[nodiscard]] ForwardIterator min_element(ExecutionPolicy&& exec,
//                                             ForwardIterator first, ForwardIterator last);
//
// template<class ExecutionPolicy, class ForwardIterator, class Compare>
//   [[nodiscard]] ForwardIterator min_element(ExecutionPolicy&& exec,
//                                             ForwardIterator first, ForwardIterator last,
//                                             Compare comp);

#include <algorithm>
#include <execution>
#include <functional>

int main(int, char**) {
  int arr[] = {1, 2, 3};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::execution::seq, arr, arr + 3);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::execution::seq, arr, arr + 3, std::less<int>());
}
