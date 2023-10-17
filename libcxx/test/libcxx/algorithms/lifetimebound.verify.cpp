//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -Wno-pessimizing-move -Wno-unused-variable

#include <algorithm>

#include "test_macros.h"

struct Comp {
  template <class T, class U>
  bool operator()(T, U) {
    return false;
  }
};

void func() {
  int i = 0;
  {
    auto&& v1 = std::min(0, i);         // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::min(i, 0);         // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::min(0, i, Comp{}); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::min(i, 0, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
  }
  {
    auto&& v1 = std::max(0, i);         // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::max(i, 0);         // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::max(0, i, Comp{}); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::max(i, 0, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
  }
  {
    auto&& v1 = std::minmax(0, i);         // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::minmax(i, 0);         // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::minmax(0, i, Comp{}); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::minmax(i, 0, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
    auto v5 = std::minmax(0, i);           // expected-warning {{temporary whose address is used as value of local variable 'v5' will be destroyed at the end of the full-expression}}
    auto v6 = std::minmax(i, 0);           // expected-warning {{temporary whose address is used as value of local variable 'v6' will be destroyed at the end of the full-expression}}
    auto v7 = std::minmax(0, i, Comp{});   // expected-warning {{temporary whose address is used as value of local variable 'v7' will be destroyed at the end of the full-expression}}
    auto v8 = std::minmax(i, 0, Comp{});   // expected-warning {{temporary whose address is used as value of local variable 'v8' will be destroyed at the end of the full-expression}}
  }
#if TEST_STD_VER >= 17
  {
    auto&& v1 = std::clamp(1, i, i); // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::clamp(i, 1, i); // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::clamp(i, i, 1); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::clamp(1, i, i, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
    auto&& v5 = std::clamp(i, 1, i, Comp{}); // expected-warning {{temporary bound to local reference 'v5' will be destroyed at the end of the full-expression}}
    auto&& v6 = std::clamp(i, i, 1, Comp{}); // expected-warning {{temporary bound to local reference 'v6' will be destroyed at the end of the full-expression}}
  }
#endif
#if TEST_STD_VER >= 20
  {
    auto&& v1 = std::ranges::min(0, i);         // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::ranges::min(i, 0);         // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::ranges::min(0, i, Comp{}); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::ranges::min(i, 0, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
  }
  {
    auto&& v1 = std::ranges::max(0, i);         // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::ranges::max(i, 0);         // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::ranges::max(0, i, Comp{}); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::ranges::max(i, 0, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
  }
  {
    auto&& v1 = std::ranges::minmax(0, i);         // expected-warning {{temporary bound to local reference 'v1' will be destroyed at the end of the full-expression}}
    auto&& v2 = std::ranges::minmax(i, 0);         // expected-warning {{temporary bound to local reference 'v2' will be destroyed at the end of the full-expression}}
    auto&& v3 = std::ranges::minmax(0, i, Comp{}); // expected-warning {{temporary bound to local reference 'v3' will be destroyed at the end of the full-expression}}
    auto&& v4 = std::ranges::minmax(i, 0, Comp{}); // expected-warning {{temporary bound to local reference 'v4' will be destroyed at the end of the full-expression}}
    auto v5 = std::ranges::minmax(0, i);           // expected-warning {{temporary whose address is used as value of local variable 'v5' will be destroyed at the end of the full-expression}}
    auto v6 = std::ranges::minmax(i, 0);           // expected-warning {{temporary whose address is used as value of local variable 'v6' will be destroyed at the end of the full-expression}}
    auto v7 = std::ranges::minmax(0, i, Comp{});   // expected-warning {{temporary whose address is used as value of local variable 'v7' will be destroyed at the end of the full-expression}}
    auto v8 = std::ranges::minmax(i, 0, Comp{});   // expected-warning {{temporary whose address is used as value of local variable 'v8' will be destroyed at the end of the full-expression}}
  }
#endif // TEST_STD_VER >= 20
}
