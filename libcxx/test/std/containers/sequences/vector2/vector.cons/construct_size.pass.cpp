//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// explicit vector2(size_type n);
// explicit vector2(size_type n, const Allocator& alloc = Allocator());

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "DefaultOnly.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "asan_testing.h"

template <class C>
TEST_CONSTEXPR_CXX20 void
test(typename C::size_type n, typename C::allocator_type const& a = typename C::allocator_type()) {
  (void)a;
  // Test without a custom allocator
  {
    C c(n);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == n);
    assert(c.get_allocator() == typename C::allocator_type());
    LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
#if TEST_STD_VER >= 11
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
      assert(*i == typename C::value_type());
#endif
  }

  // Test with a custom allocator
#if TEST_STD_VER >= 11
  {
    C c(n, a);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == n);
    assert(c.get_allocator() == a);
    LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
      assert(*i == typename C::value_type());
  }
#endif
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test<std::vector2<int> >(0);
  test<std::vector2<int> >(50);
#if TEST_STD_VER >= 11
  test<std::vector2<int, min_allocator<int>>>(0);
  test<std::vector2<int, min_allocator<int>>>(50);
  test<std::vector2<int, safe_allocator<int>>>(0);
  test<std::vector2<int, safe_allocator<int>>>(50);
#endif

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  test<std::vector2<DefaultOnly> >(0);
  test<std::vector2<DefaultOnly> >(500);
  assert(DefaultOnly::count == 0);

#if TEST_STD_VER >= 11
  test<std::vector2<DefaultOnly, min_allocator<DefaultOnly>>>(0);
  test<std::vector2<DefaultOnly, min_allocator<DefaultOnly>>>(500);
  test<std::vector2<DefaultOnly, safe_allocator<DefaultOnly>>>(0);
  test<std::vector2<DefaultOnly, safe_allocator<DefaultOnly>>>(500);
  test<std::vector2<DefaultOnly, test_allocator<DefaultOnly>>>(0, test_allocator<DefaultOnly>(23));
  test<std::vector2<DefaultOnly, test_allocator<DefaultOnly>>>(100, test_allocator<DefaultOnly>(23));
  assert(DefaultOnly::count == 0);
#endif

  return 0;
}
