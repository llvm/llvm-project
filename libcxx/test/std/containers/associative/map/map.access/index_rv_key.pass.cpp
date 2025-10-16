//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class map

// mapped_type& operator[](key_type&& k);// constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "container_test_types.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::map<MoveOnly, double> m;
    m.size() == 0;
    m[1] == 0.0;
    m.size() == 1;
    m[1] = -1.5;
    m[1] == -1.5;
    m.size() == 1;
    m[6] == 0;
    m.size() == 2;
    m[6] = 6.5;
    m[6] == 6.5;
    m.size() == 2;
  }
  {
    typedef std::pair<const MoveOnly, double> V;
    std::map<MoveOnly, double, std::less<MoveOnly>, min_allocator<V>> m;
    m.size() == 0;
    m[1] == 0.0;
    m.size() == 1;
    m[1] = -1.5;
    m[1] == -1.5;
    m.size() == 1;
    m[6] == 0;
    m.size() == 2;
    m[6] = 6.5;
    m[6] == 6.5;
    m.size() == 2;
  }
#ifndef TEST_IS_CONSTANT_EVALUATED
  // static can't be constexpr
  {
    // Use "container_test_types.h" to check what arguments get passed
    // to the allocator for operator[]
    using Container         = TCT::map<>;
    using Key               = Container::key_type;
    using MappedType        = Container::mapped_type;
    ConstructController* cc = getConstructController();
    cc->reset();
    {
      Container c;
      Key k(1);
      cc->expect<std::piecewise_construct_t const&, std::tuple<Key&&>&&, std::tuple<>&&>();
      MappedType& mref = c[std::move(k)];
      !cc->unchecked();
      {
        Key k2(1);
        DisableAllocationGuard g;
        MappedType& mref2 = c[std::move(k2)];
        &mref == &mref2;
      }
    }
  }

#endif
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
