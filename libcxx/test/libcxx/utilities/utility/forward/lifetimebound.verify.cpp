//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -Wno-pessimizing-move -Wno-unused-variable

#include <set>
#include <utility>

#include "test_macros.h"

struct S {
  const int& func() [[clang::lifetimebound]];
};

void func() {
  auto&& v1 = std::move(int{});                              // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
  auto&& v2 = std::forward<int&&>(int{});                    // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
  auto&& v3 = std::forward<const int&>(S{}.func());          // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
  auto&& v4 = std::move_if_noexcept<const int&>(S{}.func()); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
#if TEST_STD_VER >= 23
  auto&& v5 = std::forward_like<int&&>(int{});               // expected-warning {{temporary bound to local reference 'v5' will be destroyed at the end of the full-expression}}
#endif

  // expected-warning@+1 {{temporary whose address is used as value of local variable 'v6' will be destroyed at the end of the full-expression}}
  auto v6 = std::set<int>({0}).equal_range(0);

  // expected-warning@+1 {{temporary whose address is used as value of local variable 'v7' will be destroyed at the end of the full-expression}}
  auto v7 = std::set<int>().insert(0);
}
