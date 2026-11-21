//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// vector2(vector2&&)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value);

// This tests a conforming extension


#include <vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_alloc {
  typedef T value_type;
  some_alloc(const some_alloc&);
  void allocate(std::size_t);
};

int main(int, char**) {
  {
    typedef std::vector2<MoveOnly> C;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }
  {
    typedef std::vector2<MoveOnly, test_allocator<MoveOnly>> C;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }
  {
    typedef std::vector2<MoveOnly, other_allocator<MoveOnly>> C;
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
  }
  {
    typedef std::vector2<MoveOnly, some_alloc<MoveOnly>> C;
    //  In C++17, move constructors for allocators are not allowed to throw
#if TEST_STD_VER > 14
    static_assert(std::is_nothrow_move_constructible<C>::value, "");
#else
    static_assert(!std::is_nothrow_move_constructible<C>::value, "");
#endif
  }

  return 0;
}
