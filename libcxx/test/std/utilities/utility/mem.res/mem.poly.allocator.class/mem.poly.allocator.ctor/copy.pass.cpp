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

// <memory_resource>

// template <class T> class polymorphic_allocator

// polymorphic_allocator<T>::polymorphic_allocator(polymorphic_allocator const &);

#include <memory_resource>
#include <cassert>
#include <type_traits>
#include <utility>

int main(int, char**) {
  typedef std::pmr::polymorphic_allocator<void> A1;
  {
    static_assert(std::is_copy_constructible<A1>::value, "");
    static_assert(std::is_move_constructible<A1>::value, "");
  }
  // copy
  {
    A1 const a((std::pmr::memory_resource*)42);
    A1 const a2(a);
    assert(a.resource() == a2.resource());
  }
  // move
  {
    A1 a((std::pmr::memory_resource*)42);
    A1 a2(std::move(a));
    assert(a.resource() == a2.resource());
    assert(a2.resource() == (std::pmr::memory_resource*)42);
  }

  return 0;
}
