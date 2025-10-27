//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(const basic_string& str, const Allocator& alloc); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
struct alloc_imp {
  bool active;

  TEST_CONSTEXPR alloc_imp() : active(true) {}

  template <class T>
  T* allocate(std::size_t n) {
    if (active)
      return static_cast<T*>(std::malloc(n * sizeof(T)));
    else
      throw std::bad_alloc();
  }

  template <class T>
  void deallocate(T* p, std::size_t) {
    std::free(p);
  }
  void activate() { active = true; }
  void deactivate() { active = false; }
};

template <class T>
struct poca_alloc {
  typedef T value_type;
  typedef std::true_type propagate_on_container_copy_assignment;

  alloc_imp* imp;

  TEST_CONSTEXPR poca_alloc(alloc_imp* imp_) : imp(imp_) {}

  template <class U>
  TEST_CONSTEXPR poca_alloc(const poca_alloc<U>& other) : imp(other.imp) {}

  T* allocate(std::size_t n) { return imp->allocate<T>(n); }
  void deallocate(T* p, std::size_t n) { imp->deallocate(p, n); }
};

template <typename T, typename U>
bool operator==(const poca_alloc<T>& lhs, const poca_alloc<U>& rhs) {
  return lhs.imp == rhs.imp;
}

template <typename T, typename U>
bool operator!=(const poca_alloc<T>& lhs, const poca_alloc<U>& rhs) {
  return lhs.imp != rhs.imp;
}
#endif

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s1, const typename S::allocator_type& a) {
  S s2(s1, a);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2 == s1);
  assert(s2.capacity() >= s2.size());
  assert(s2.get_allocator() == a);
  LIBCPP_ASSERT(is_string_asan_correct(s1));
  LIBCPP_ASSERT(is_string_asan_correct(s2));
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test_string(const Alloc& a) {
  typedef std::basic_string<char, std::char_traits<char>, Alloc> S;
  test(S(), Alloc(a));
  test(S("1"), Alloc(a));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890"), Alloc(a));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string(std::allocator<char>());
  test_string(test_allocator<char>());
  test_string(test_allocator<char>(3));
#if TEST_STD_VER >= 11
  test_string(min_allocator<char>());
  test_string(safe_allocator<char>());
#endif

#if TEST_STD_VER >= 11
#  ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    typedef poca_alloc<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
    const char* p1 = "This is my first string";
    const char* p2 = "This is my second string";

    alloc_imp imp1;
    alloc_imp imp2;
    S s1(p1, A(&imp1));
    S s2(p2, A(&imp2));

    assert(s1 == p1);
    assert(s2 == p2);

    imp2.deactivate();
    try {
      s1 = s2;
      assert(false);
    } catch (std::bad_alloc&) {
    }
    assert(s1 == p1);
    assert(s2 == p2);
  }
#  endif
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
