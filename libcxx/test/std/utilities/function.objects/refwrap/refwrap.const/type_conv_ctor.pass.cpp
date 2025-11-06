//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&) noexcept(see below);

#include <functional>
#include <cassert>

#include "test_macros.h"

struct convertible_to_int_ref {
  int val = 0;
  operator int&() { return val; }
  operator int const&() const { return val; }
};

template <bool IsNothrow>
struct nothrow_convertible {
  int val = 0;
  operator int&() TEST_NOEXCEPT_COND(IsNothrow) { return val; }
};

struct convertible_from_int {
  convertible_from_int(int) {}
};

void meow(std::reference_wrapper<int>) {}
void meow(convertible_from_int) {}

std::reference_wrapper<int> purr();

int main(int, char**) {
  {
    convertible_to_int_ref t;
    std::reference_wrapper<convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    const convertible_to_int_ref t;
    std::reference_wrapper<const convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    using Ref = std::reference_wrapper<int>;
    ASSERT_NOEXCEPT(Ref(nothrow_convertible<true>()));
    ASSERT_NOT_NOEXCEPT(Ref(nothrow_convertible<false>()));
  }
  meow(0);
  ASSERT_SAME_TYPE(decltype(true ? purr() : 0), int);
#if TEST_STD_VER > 14
  {
    int i = 0;
    std::reference_wrapper ri(i);
    static_assert((std::is_same<decltype(ri), std::reference_wrapper<int>>::value), "");
  }
  {
    const int j = 0;
    std::reference_wrapper rj(j);
    static_assert((std::is_same<decltype(rj), std::reference_wrapper<const int>>::value), "");
  }
#endif

  return 0;
}
