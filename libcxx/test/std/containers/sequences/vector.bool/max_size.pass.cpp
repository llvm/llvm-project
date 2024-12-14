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
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "min_allocator.h"
#include "sized_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11

template <typename Alloc>
TEST_CONSTEXPR_CXX20 void test(const std::vector<bool, Alloc>& v) {
  using Vector              = std::vector<bool, Alloc>;
  using size_type           = typename Vector::size_type;
  using difference_type     = typename Vector::difference_type;
  using storage_type        = typename Vector::__storage_type;
  using storage_alloc       = typename std::allocator_traits<Alloc>::template rebind_alloc<storage_type>;
  using storage_traits      = typename std::allocator_traits<Alloc>::template rebind_traits<storage_type>;
  const size_type max_dist  = static_cast<size_type>(std::numeric_limits<difference_type>::max());
  const size_type max_alloc = storage_traits::max_size(storage_alloc(v.get_allocator()));
  std::size_t bits_per_word = sizeof(storage_type) * CHAR_BIT;
  const size_type max_size  = max_dist / bits_per_word < max_alloc ? max_dist : max_alloc * bits_per_word;
  assert(v.max_size() <= max_dist);
  assert(v.max_size() / bits_per_word <= max_alloc); // max_alloc * bits_per_word may overflow
  LIBCPP_ASSERT(v.max_size() == max_size);
}

#endif

TEST_CONSTEXPR_CXX20 bool tests() {
  // Test with limited_allocator where v.max_size() is determined by allocator::max_size()
  {
    using Alloc        = limited_allocator<bool, 10>;
    using Vector       = std::vector<bool, Alloc>;
    using storage_type = Vector::__storage_type;
    Vector v;
    std::size_t bits_per_word = sizeof(storage_type) * CHAR_BIT;
    assert(v.max_size() <= 10 * bits_per_word);
    LIBCPP_ASSERT(v.max_size() == 10 * bits_per_word);
  }
  {
    using Alloc        = limited_allocator<double, 10>;
    using Vector       = std::vector<bool, Alloc>;
    using storage_type = Vector::__storage_type;
    Vector v;
    std::size_t bits_per_word = sizeof(storage_type) * CHAR_BIT;
    assert(v.max_size() <= 10 * bits_per_word);
    LIBCPP_ASSERT(v.max_size() == 10 * bits_per_word);
  }

#if TEST_STD_VER >= 11

  // Test with various allocators and diffrent size_type
  {
    test(std::vector<bool>());
    test(std::vector<bool, std::allocator<int> >());
    test(std::vector<bool, min_allocator<bool> >());
    test(std::vector<bool, test_allocator<bool> >(test_allocator<bool>(1)));
    test(std::vector<bool, other_allocator<bool> >(other_allocator<bool>(5)));
    test(std::vector<bool, sized_allocator<bool, std::uint8_t, std::int8_t> >());
    test(std::vector<bool, sized_allocator<bool, std::uint16_t, std::int16_t> >());
    test(std::vector<bool, sized_allocator<bool, std::uint32_t, std::int32_t> >());
    test(std::vector<bool, sized_allocator<bool, std::uint64_t, std::int64_t> >());
    test(std::vector<bool, limited_allocator<bool, static_cast<std::size_t>(-1)> >());
  }

  // Test cases to identify incorrect implementations that unconditionally return:
  //   std::min<size_type>(__nmax, __internal_cap_to_external(__amax))
  // This can cause overflow in __internal_cap_to_external and lead to incorrect results.
  {
    test(std::vector<bool, limited_allocator<bool, static_cast<std::size_t>(-1) / 61> >());
    test(std::vector<bool, limited_allocator<bool, static_cast<std::size_t>(-1) / 63> >());
  }

#endif

  return true;
}

int main(int, char**) {
  tests();

#if TEST_STD_VER >= 20
  static_assert(tests());
#endif

  return 0;
}
