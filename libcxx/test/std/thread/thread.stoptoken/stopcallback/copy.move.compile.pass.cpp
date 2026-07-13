//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

//  stop_callback(const stop_callback&) = delete;
//  stop_callback(stop_callback&&) = delete;
//  stop_callback& operator=(const stop_callback&) = delete;
//  stop_callback& operator=(stop_callback&&) = delete;

#include <stop_token>
#include <type_traits>

struct Callback {
  void operator()() const;
};

static_assert(!std::is_copy_constructible_v<std::stop_callback<Callback>>);
static_assert(!std::is_move_constructible_v<std::stop_callback<Callback>>);
static_assert(!std::is_copy_assignable_v<std::stop_callback<Callback>>);
static_assert(!std::is_move_assignable_v<std::stop_callback<Callback>>);
