//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// template<class E> class initializer_list;

// const E* begin() const noexcept; // constexpr since C++14
// const E* end() const noexcept;   // constexpr since C++14
// const E* data() const noexcept;  // constexpr since C++14
// size_t size() const noexcept;    // constexpr since C++14
// bool empty() const noexcept;     // constexpr since C++14

#include <initializer_list>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

ASSERT_SAME_TYPE(decltype(std::initializer_list<int>{}.begin()), const int*);
ASSERT_SAME_TYPE(decltype(std::initializer_list<int>{}.end()), const int*);
ASSERT_SAME_TYPE(decltype(std::initializer_list<int>{}.data()), const int*);
ASSERT_SAME_TYPE(decltype(std::initializer_list<int>{}.size()), std::size_t);
ASSERT_SAME_TYPE(decltype(std::initializer_list<int>{}.empty()), bool);

ASSERT_NOEXCEPT(std::initializer_list<int>{}.begin());
ASSERT_NOEXCEPT(std::initializer_list<int>{}.end());
ASSERT_NOEXCEPT(std::initializer_list<int>{}.data());
ASSERT_NOEXCEPT(std::initializer_list<int>{}.size());
ASSERT_NOEXCEPT(std::initializer_list<int>{}.empty());

struct A
{
  TEST_CONSTEXPR_CXX14 A(std::initializer_list<int> il) {
    {
      const int* b = il.begin();
      const int* e = il.end();
      assert(il.data() == b);
      assert(il.size() == 3);
      assert(!il.empty());
      assert(static_cast<std::size_t>(e - b) == il.size());
      assert(*b++ == 3);
      assert(*b++ == 2);
      assert(*b++ == 1);
    }
    {
      const auto cil = il;
      const int* b   = cil.begin();
      const int* e   = cil.end();
      assert(cil.data() == b);
      assert(cil.size() == 3);
      assert(!cil.empty());
      assert(static_cast<std::size_t>(e - b) == cil.size());
      assert(*b++ == 3);
      assert(*b++ == 2);
      assert(*b++ == 1);
    }
  }
};

TEST_CONSTEXPR_CXX14 bool test_empty_ilist() {
  {
    std::initializer_list<int> il{};
    const int* b = il.begin();
    const int* e = il.end();
    assert(il.data() == b);
    assert(il.size() == 0);
    assert(il.empty());
    assert(static_cast<std::size_t>(e - b) == il.size());
  }
  {
    const std::initializer_list<int> cil{};
    const int* b = cil.begin();
    const int* e = cil.end();
    assert(cil.data() == b);
    assert(cil.size() == 0);
    assert(cil.empty());
    assert(static_cast<std::size_t>(e - b) == cil.size());
  }

  return true;
}

int main(int, char**)
{
    A test1 = {3, 2, 1}; (void)test1;
    test_empty_ilist();
#if TEST_STD_VER > 11
    constexpr A test2 = {3, 2, 1};
    (void)test2;
    static_assert(test_empty_ilist(), "");
#endif // TEST_STD_VER > 11

  return 0;
}
