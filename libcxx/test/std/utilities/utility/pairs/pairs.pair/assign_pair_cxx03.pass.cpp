//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"

struct NonAssignable {
  NonAssignable() {}
private:
  NonAssignable& operator=(NonAssignable const&);
};

struct Incomplete;
extern Incomplete inc_obj;

struct ConstructibleFromInt {
  ConstructibleFromInt() : value(-1) { }
  explicit ConstructibleFromInt(int v) : value(v) { }
  int value;
};

int main(int, char**)
{
    {
      // Test that we don't constrain the assignment operator in C++03 mode.
      // Since we don't have access control SFINAE having pair evaluate SFINAE
      // may cause a hard error.
      typedef std::pair<int, NonAssignable> P;
      static_assert(std::is_copy_assignable<P>::value, "");
    }
    {
      typedef std::pair<int, Incomplete&> P;
      static_assert(std::is_copy_assignable<P>::value, "");
      P p(42, inc_obj);
      assert(&p.second == &inc_obj);
    }
    {
      // The type is constructible from int, but not assignable from int.
      // This ensures that operator=(pair const&) can be used in conjunction with
      // pair(pair<U, V> const&) to mimic operator=(pair<U, V> const&) in C++03.
      // This is weird but valid in C++03.
      std::pair<ConstructibleFromInt, char> p;
      std::pair<int, char> from(11, 'x');
      p = from;
      assert(p.first.value == 11);
      assert(p.second == 'x');
    }

  return 0;
}

struct Incomplete {};
Incomplete inc_obj;
