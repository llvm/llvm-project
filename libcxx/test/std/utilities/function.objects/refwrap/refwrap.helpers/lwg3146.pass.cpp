//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<T> ref(reference_wrapper<T> t);
//   LWG 3146 "Excessive unwrapping in std::ref/cref"

#include <functional>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test()
{
  {
    int i = 0;
    std::reference_wrapper<int> ri(i);
    std::reference_wrapper<std::reference_wrapper<int> > rri(ri);
    auto rrj = std::ref(rri);
    ASSERT_SAME_TYPE(decltype(rrj), std::reference_wrapper<std::reference_wrapper<int> >);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    std::reference_wrapper<int> ri(i);
    std::reference_wrapper<const std::reference_wrapper<int> > rri(ri);
    auto rrj = std::ref(rri);
    ASSERT_SAME_TYPE(decltype(rrj), std::reference_wrapper<const std::reference_wrapper<int> >);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    std::reference_wrapper<int> ri(i);
    std::reference_wrapper<std::reference_wrapper<int> > rri(ri);
    auto rrj = std::cref(rri);
    ASSERT_SAME_TYPE(decltype(rrj), std::reference_wrapper<const std::reference_wrapper<int> >);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    std::reference_wrapper<int> ri(i);
    std::reference_wrapper<const std::reference_wrapper<int> > rri(ri);
    auto rrj = std::cref(rri);
    ASSERT_SAME_TYPE(decltype(rrj), std::reference_wrapper<const std::reference_wrapper<int> >);
    assert(&rrj.get() == &ri);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
