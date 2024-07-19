//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb
// ADDITIONAL_COMPILE_FLAGS(clang): -Wno-deprecated
// ADDITIONAL_COMPILE_FLAGS(gcc): -Wno-deprecated-copy-dtor

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class ambiguous_local_time : public runtime_error {
// public:
//   template<class Duration>
//     ambiguous_local_time(const local_time<Duration>& tp, const local_info& i);
// };

#include <chrono>
#include <stdexcept>
#include <type_traits>

// Basic properties
static_assert(std::is_base_of_v<std::runtime_error, std::chrono::ambiguous_local_time>);
static_assert(!std::is_default_constructible_v<std::chrono::ambiguous_local_time>);
static_assert(std::is_destructible_v<std::chrono::ambiguous_local_time>);
static_assert(std::is_copy_constructible_v<std::chrono::ambiguous_local_time>);
static_assert(std::is_move_constructible_v<std::chrono::ambiguous_local_time>);
static_assert(std::is_copy_assignable_v<std::chrono::ambiguous_local_time>);
static_assert(std::is_move_assignable_v<std::chrono::ambiguous_local_time>);

int main(int, char**) {
  std::chrono::ambiguous_local_time e{
      std::chrono::local_seconds{}, std::chrono::local_info{std::chrono::local_info::ambiguous, {}, {}}};

  std::chrono::ambiguous_local_time copy = e;
  copy                                   = e;

  std::chrono::ambiguous_local_time move = std::move(e);
  e                                      = move;
  move                                   = std::move(e);

  return 0;
}
