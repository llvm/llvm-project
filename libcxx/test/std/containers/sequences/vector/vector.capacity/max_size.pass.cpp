//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// size_type max_size() const;

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "min_allocator.h"
#include "sized_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11

template <typename T, typename Alloc>
TEST_CONSTEXPR_CXX20 void test(const std::vector<T, Alloc>& v) {
  using Vector              = std::vector<T, Alloc>;
  using alloc_traits        = std::allocator_traits<typename Vector::allocator_type>;
  using size_type           = typename Vector::size_type;
  using difference_type     = typename Vector::difference_type;
  const size_type max_dist  = static_cast<size_type>(std::numeric_limits<difference_type>::max());
  const size_type max_alloc = alloc_traits::max_size(v.get_allocator());
  assert(v.max_size() <= max_dist);
  assert(v.max_size() <= max_alloc);
  LIBCPP_ASSERT(v.max_size() == std::min<size_type>(max_dist, max_alloc));
}

#endif // TEST_STD_VER >= 11

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    typedef limited_allocator<int, 10> A;
    typedef std::vector<int, A> C;
    C c;
    assert(c.max_size() <= 10);
    LIBCPP_ASSERT(c.max_size() == 10);
  }
  {
    typedef limited_allocator<int, (std::size_t)-1> A;
    typedef std::vector<int, A> C;
    const C::size_type max_dist = static_cast<C::size_type>(std::numeric_limits<C::difference_type>::max());
    C c;
    assert(c.max_size() <= max_dist);
    LIBCPP_ASSERT(c.max_size() == max_dist);
  }
  {
    typedef std::vector<char> C;
    const C::size_type max_dist = static_cast<C::size_type>(std::numeric_limits<C::difference_type>::max());
    C c;
    assert(c.max_size() <= max_dist);
    assert(c.max_size() <= alloc_max_size(c.get_allocator()));
    LIBCPP_ASSERT(c.max_size() == std::min(max_dist, alloc_max_size(c.get_allocator())));
  }

#if TEST_STD_VER >= 11

  // Test with various allocators and diffrent size_type
  {
    test(std::vector<int>());
    test(std::vector<short, std::allocator<short> >());
    test(std::vector<unsigned, min_allocator<unsigned> >());
    test(std::vector<char, test_allocator<char> >(test_allocator<char>(1)));
    test(std::vector<std::size_t, other_allocator<std::size_t> >(other_allocator<std::size_t>(5)));
    test(std::vector<int, sized_allocator<int, std::uint8_t, std::int8_t> >());
    test(std::vector<int, sized_allocator<int, std::uint16_t, std::int16_t> >());
    test(std::vector<int, sized_allocator<int, std::uint32_t, std::int32_t> >());
    test(std::vector<int, sized_allocator<int, std::uint64_t, std::int64_t> >());
    test(std::vector<int, limited_allocator<int, static_cast<std::size_t>(-1)> >());
    test(std::vector<int, limited_allocator<int, static_cast<std::size_t>(-1) / 2> >());
    test(std::vector<int, limited_allocator<int, static_cast<std::size_t>(-1) / 4> >());
  }

#endif // TEST_STD_VER >= 11

  return true;
}

int main(int, char**) {
  tests();

#if TEST_STD_VER > 17
  static_assert(tests());
#endif

  return 0;
}
