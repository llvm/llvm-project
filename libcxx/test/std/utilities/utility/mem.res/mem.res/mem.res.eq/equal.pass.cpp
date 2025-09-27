//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

// bool operator==(memory_resource const &, memory_resource const &) noexcept;

#include <memory_resource>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_std_memory_resource.h"

int main(int, char**) {
  // check return types
  {
    const std::pmr::memory_resource* mr1 = nullptr;
    const std::pmr::memory_resource* mr2 = nullptr;
    ASSERT_SAME_TYPE(decltype(*mr1 == *mr2), bool);
    ASSERT_NOEXCEPT(*mr1 == *mr2);
  }
  // equal
  {
    TestResource r1(1);
    TestResource r2(1);
    const std::pmr::memory_resource& mr1 = r1;
    const std::pmr::memory_resource& mr2 = r2;

    assert(mr1 == mr2);
    assert(r1.checkIsEqualCalledEq(1));
    assert(r2.checkIsEqualCalledEq(0));

    assert(mr2 == mr1);
    assert(r1.checkIsEqualCalledEq(1));
    assert(r2.checkIsEqualCalledEq(1));
  }
  // equal same object
  {
    TestResource r1(1);
    const std::pmr::memory_resource& mr1 = r1;
    const std::pmr::memory_resource& mr2 = r1;

    assert(mr1 == mr2);
    assert(r1.checkIsEqualCalledEq(0));

    assert(mr2 == mr1);
    assert(r1.checkIsEqualCalledEq(0));
  }
  // not equal
  {
    TestResource r1(1);
    TestResource r2(2);
    const std::pmr::memory_resource& mr1 = r1;
    const std::pmr::memory_resource& mr2 = r2;

    assert(!(mr1 == mr2));
    assert(r1.checkIsEqualCalledEq(1));
    assert(r2.checkIsEqualCalledEq(0));

    assert(!(mr2 == mr1));
    assert(r1.checkIsEqualCalledEq(1));
    assert(r2.checkIsEqualCalledEq(1));
  }

  return 0;
}
