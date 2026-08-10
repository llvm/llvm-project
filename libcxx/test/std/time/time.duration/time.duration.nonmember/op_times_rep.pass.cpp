//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const duration<Rep1, Period>& d, const Rep2& s);

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const Rep1& s, const duration<Rep2, Period>& d);

#include <chrono>
#include <cassert>

#include "test_macros.h"
#include "../../rep.h"

int main(int, char**) {
  {
    std::chrono::nanoseconds ns(3);
    ns = ns * 5;
    assert(ns.count() == 15);
    ns = 6 * ns;
    assert(ns.count() == 90);
  }

#if TEST_STD_VER >= 11
  {
    constexpr std::chrono::nanoseconds ns(3);
    constexpr std::chrono::nanoseconds ns2 = ns * 5;
    static_assert(ns2.count() == 15, "");
    constexpr std::chrono::nanoseconds ns3 = 6 * ns;
    static_assert(ns3.count() == 18, "");
  }
#endif

#if TEST_STD_VER >= 11
  { // This is related to PR#41130
    typedef std::chrono::nanoseconds Duration;
    Duration d(5);
    NotARep n;
    ASSERT_SAME_TYPE(Duration, decltype(d * n));
    ASSERT_SAME_TYPE(Duration, decltype(n * d));
    d = d * n;
    assert(d.count() == 5);
    d = n * d;
    assert(d.count() == 5);
  }
  {
    std::chrono::duration<int> d(8);
    RepConstConvertibleLWG3050 x;

    {
      auto r = d * x;
      assert(r.count() == 16);
      ASSERT_SAME_TYPE(std::chrono::duration<long>, decltype(r));
    }
    {
      auto r = x * d;
      assert(r.count() == 16);
      ASSERT_SAME_TYPE(std::chrono::duration<long>, decltype(r));
    }
  }
#endif // TEST_STD_VER >= 11

  return 0;
}
