//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <memory>

#include <cassert>
#include <memory>

//#include "test_macros.h"

template <bool Expected> struct test_helper {  
  template <typename Lmbd> auto & operator=(Lmbd && lmbd) {
    if constexpr (Expected) {
      static_assert(Lmbd{}());
      assert(lmbd());
    } else {
      static_assert(!Lmbd{}());
      assert(!lmbd());
    }
    
    return *this;
  }
};

#define HANA_TEST test_helper<true>{} = []
#define HANA_FAIL_TEST test_helper<false>{} = []
#define HANA_ASSERT(expr) do { if (!static_cast<bool>(expr)) { if consteval { throw false; } return false; }} while (false)

template <typename T> struct tagged_range {
  using scheme = std::memory::no_tag;
  using iterator = std::tagged_ptr<const T, bool, scheme>;
  
  const T * start;
  size_t size;
  
  constexpr tagged_range(const T * in, std::size_t sz) noexcept: start{in}, size{sz} { }
  
  constexpr auto begin() const {
    return iterator(start, false);
  }
  
  constexpr auto end() const {
    return iterator(start+size, false);
  }
};

int main(int, char**) {
  
HANA_TEST {
  int64_t a = 42;

  uintptr_t tag = 0b11u;
  auto tptr = std::tagged_ptr<int64_t, uintptr_t, std::memory::alignment_low_bits_tag>(&a, tag);

  HANA_ASSERT(tptr.unsafe_dirty_pointer() != &a);

  int64_t * original = tptr.pointer();
  HANA_ASSERT(tag == tptr.tag());
  HANA_ASSERT(original == &a);

  auto [p, t] = tptr;
  HANA_ASSERT(p == &a);
  HANA_ASSERT(t == tag);
  return true;
};

HANA_TEST {
  int64_t a[3] = {1,2,3};
 
  uintptr_t tag = 0b11000111u;
  auto tptr = std::tagged_ptr<int64_t[3], uintptr_t, std::memory::low_byte_tag>(&a, tag);
 
  auto * original = tptr.pointer();
  HANA_ASSERT(tag == tptr.tag());
  HANA_ASSERT(original == &a);
 
  auto [p, t] = tptr;
  HANA_ASSERT(p == &a);
  HANA_ASSERT(t == tag);
 
  HANA_ASSERT(tptr[0] == 1);
  HANA_ASSERT(tptr[1] == 2);
  HANA_ASSERT(tptr[2] == 3);
 
 
  return true;
};
  
  
HANA_TEST {
  int64_t array[8] = {1,2,3,4,5,6,7,8};
  int64_t * ptr = &array[0];
  auto tptr = std::tagged_ptr<int64_t, unsigned, std::memory::alignment_low_bits_tag>{ptr, 0b1u};
  int64_t sum = 0;
  while (tptr != &array[8]) {
    sum += *tptr;
    ++tptr;
  }
  return sum;
};

HANA_TEST {
  int64_t array[8] = {1,2,3,4,5,6,7,8};
  auto rng = tagged_range(&array[0], 8);
  
  static_assert(std::input_or_output_iterator<decltype(rng.begin())>);
  static_assert(std::input_iterator<decltype(rng.begin())>);
  static_assert(std::forward_iterator<decltype(rng.begin())>);
  static_assert(std::bidirectional_iterator<decltype(rng.begin())>);
  static_assert(std::random_access_iterator<decltype(rng.begin())>);
  //static_assert(std::contiguous_iterator<decltype(rng.begin())>);
  
  static_assert(std::ranges::input_range<decltype(rng)>);
  static_assert(std::ranges::forward_range<decltype(rng)>);
  static_assert(std::ranges::bidirectional_range<decltype(rng)>);
  static_assert(std::ranges::random_access_range<decltype(rng)>);
  //static_assert(std::ranges::contiguous_range<decltype(rng)>);
  //static_assert(std::forward_iterator<tagged_range<int64_t>>);
  //static_assert(std::ranges::bidirectional_<tagged_range<int64_t>>);
  
  int64_t sum = 0;
  for (int64_t v: rng) {
    sum += v;
  }
  return sum;
};

HANA_TEST {
  int a{42};
  auto tptr = std::tagged_ptr<int, bool, std::memory::alignment_low_bits_tag>{&a, false};
  
  auto cptr = std::const_pointer_cast<const int>(tptr);
  
  HANA_ASSERT(cptr.tag() == false);
  HANA_ASSERT(cptr.pointer() == &a);
  
  return true;
};

HANA_TEST {
  int a{14};
  int * b = &a;
  auto tptr = std::tagged_ptr<int, uintptr_t, std::memory::alignment_low_bits_tag>(b);
  HANA_ASSERT(tptr.pointer() == &a);
  return true;
};



  
  return 0;
}

