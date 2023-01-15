//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// XFAIL: clang-14, clang-15, gcc-12, apple-clang-14

// checks that CTAD for std::function works properly with static operator() overloads

#include <functional>
#include <type_traits>

struct Except0 {
  static void operator()() {}
};
static_assert(std::is_same_v<decltype(std::function{Except0{}}), std::function<void()>>);

struct Except1 {
  static int operator()(int&) { return 0; }
};
static_assert(std::is_same_v<decltype(std::function{Except1{}}), std::function<int(int&)>>);

struct Except2 {
  static int operator()(int*, long*) { return 0; }
};
static_assert(std::is_same_v<decltype(std::function{Except2{}}), std::function<int(int*, long*)>>);

struct ExceptD {
  static int operator()(int*, long* = nullptr) { return 0; }
};
static_assert(std::is_same_v<decltype(std::function{ExceptD{}}), std::function<int(int*, long*)>>);

struct Noexcept {
  static int operator()(int*, long*) noexcept { return 0; }
};

static_assert(std::is_same_v<decltype(std::function{Noexcept{}}), std::function<int(int*, long*)>>);
