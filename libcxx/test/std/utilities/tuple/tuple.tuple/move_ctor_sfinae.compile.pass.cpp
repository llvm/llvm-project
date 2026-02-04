//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that tuple's move constructor properly SFINAES.
// This is a regression test for https://github.com/llvm/llvm-project/pull/151654#issuecomment-3205410955

// UNSUPPORTED: c++03, c++11, c++14

#include <tuple>
#include <variant>
#include <type_traits>

struct S {
  S(const S&)            = delete;
  S& operator=(const S&) = delete;
  S(S&&)                 = default;
  S& operator=(S&&)      = default;
};

using T = std::tuple<const std::variant<S>>;

void func() { (void)std::is_trivially_move_constructible<T>::value; }
