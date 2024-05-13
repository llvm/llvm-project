//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17

//  atomic(const atomic&) = delete;
//  atomic& operator=(const atomic&) = delete;
//  atomic& operator=(const atomic&) volatile = delete;

#include <atomic>
#include <type_traits>

template <class T>
void test() {
  static_assert(!std::is_copy_assignable_v<std::atomic<T>>);
  static_assert(!std::is_copy_constructible_v<std::atomic<T>>);
  static_assert(!std::is_move_constructible_v<std::atomic<T>>);
  static_assert(!std::is_move_assignable_v<std::atomic<T>>);
  static_assert(!std::is_assignable_v<volatile std::atomic<T>&, const std::atomic<T>&>);
}

template void test<float>();
template void test<double>();
template void test<long double>();
