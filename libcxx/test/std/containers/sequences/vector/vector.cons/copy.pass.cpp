//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(const vector& v);

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

#if TEST_STD_VER >= 11
namespace gh196645 {

class Optional {
public:
  Optional()                           = default;
  Optional(const Optional&)            = default;
  Optional& operator=(const Optional&) = default;
  Optional(Optional&&) {}

  template <class Arg>
  Optional& operator=(Arg&& rhs) {
    assert(value_ != Poison);
    value_ = rhs.value_;
    return *this;
  }

  TEST_CONSTEXPR_CXX20 bool hasValue() const { return value_ == HasValue; }

private:
  static constexpr std::uint32_t Poison   = 0xBEBEBEBE;
  static constexpr std::uint32_t NoValue  = 0;
  static constexpr std::uint32_t HasValue = 1;

  std::uint32_t value_ = NoValue;
};

static_assert(std::is_trivially_copy_constructible<Optional>::value, "");
static_assert(std::is_trivially_copy_assignable<Optional>::value, "");
static_assert(!std::is_trivially_assignable<Optional&, Optional&>::value, "");

template <class T>
struct PoisoningAllocator {
  using value_type = T;

  TEST_CONSTEXPR_CXX20 PoisoningAllocator() = default;

  template <class U>
  TEST_CONSTEXPR_CXX20 PoisoningAllocator(const PoisoningAllocator<U>&) noexcept {}

  TEST_CONSTEXPR_CXX20 T* allocate(std::size_t n) {
    T* p = alloc_.allocate(n);
    if (!TEST_IS_CONSTANT_EVALUATED)
      std::memset(static_cast<void*>(p), 0xBE, n * sizeof(T));
    return p;
  }

  TEST_CONSTEXPR_CXX20 void deallocate(T* p, std::size_t n) noexcept { alloc_.deallocate(p, n); }

  template <class U>
  friend bool operator==(const PoisoningAllocator&, const PoisoningAllocator<U>&) noexcept {
    return true;
  }

  template <class U>
  friend bool operator!=(const PoisoningAllocator&, const PoisoningAllocator<U>&) noexcept {
    return false;
  }

  std::allocator<T> alloc_;
};

} // namespace gh196645
#endif // TEST_STD_VER >= 11

template <class C>
TEST_CONSTEXPR_CXX20 void test(const C& x) {
  typename C::size_type s = x.size();
  C c(x);
  LIBCPP_ASSERT(c.__invariants());
  assert(c.size() == s);
  assert(c == x);
  LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
}

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
    int* an = a + sizeof(a) / sizeof(a[0]);
    test(std::vector<int>(a, an));
  }
  {
    std::vector<int, test_allocator<int> > v(3, 2, test_allocator<int>(5));
    std::vector<int, test_allocator<int> > v2 = v;
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
    assert(v2 == v);
    assert(v2.get_allocator() == v.get_allocator());
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
  }
  {
    // Test copy ctor with empty source
    std::vector<int, test_allocator<int> > v(test_allocator<int>(5));
    std::vector<int, test_allocator<int> > v2 = v;
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
    assert(v2 == v);
    assert(v2.get_allocator() == v.get_allocator());
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
    assert(v2.empty());
  }
#if TEST_STD_VER >= 11
  {
    std::vector<int, other_allocator<int> > v(3, 2, other_allocator<int>(5));
    std::vector<int, other_allocator<int> > v2 = v;
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
    assert(v2 == v);
    assert(v2.get_allocator() == other_allocator<int>(-2));
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
  }
  {
    int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
    int* an = a + sizeof(a) / sizeof(a[0]);
    test(std::vector<int, min_allocator<int>>(a, an));
    test(std::vector<int, safe_allocator<int>>(a, an));
  }
  {
    std::vector<int, min_allocator<int> > v(3, 2, min_allocator<int>());
    std::vector<int, min_allocator<int> > v2 = v;
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
    assert(v2 == v);
    assert(v2.get_allocator() == v.get_allocator());
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
  }
  {
    std::vector<int, safe_allocator<int> > v(3, 2, safe_allocator<int>());
    std::vector<int, safe_allocator<int> > v2 = v;
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
    assert(v2 == v);
    assert(v2.get_allocator() == v.get_allocator());
    assert(is_contiguous_container_asan_correct(v));
    assert(is_contiguous_container_asan_correct(v2));
  }
  {
    std::vector<gh196645::Optional, gh196645::PoisoningAllocator<gh196645::Optional> > v(1);
    std::vector<gh196645::Optional, gh196645::PoisoningAllocator<gh196645::Optional> > v2 = v;
    assert(!v2[0].hasValue());
  }
#endif

  return true;
}

void test_copy_from_volatile_src() {
  volatile int src[] = {1, 2, 3};
  std::vector<int> v(src, src + 3);
  assert(v[0] == 1);
  assert(v[1] == 2);
  assert(v[2] == 3);
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  test_copy_from_volatile_src();
  return 0;
}
