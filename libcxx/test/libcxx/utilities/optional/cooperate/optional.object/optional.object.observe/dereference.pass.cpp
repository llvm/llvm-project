//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T& optional<T>::operator*() &;

#include <optional>
#include <type_traits>
#include <functional>
#include <cassert>

#include "test_macros.h"

using std::optional;

template <typename T, typename G>
constexpr auto test(T init, G test) {
  {
    optional<T> opt;
    (void)opt;
    ASSERT_SAME_TYPE(decltype(*opt), T&);
    LIBCPP_STATIC_ASSERT(noexcept(*opt));
  }

  optional<T> opt(init);
  return test(*opt);
}

int main(int, char**) {
  {
    static int i;
    assert(test<std::reference_wrapper<int>>(i, [](std::reference_wrapper<int>& r) { return &r.get(); }) == &i);
#if TEST_STD_VER > 17
    static_assert(test<std::reference_wrapper<int>>(i, [](std::reference_wrapper<int>& r) { return &r.get(); }) == &i);
#endif
  }
}
