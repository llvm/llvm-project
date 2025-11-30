//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

// constexpr indirect(const indirect& other)

// constexpr indirect(allocator_arg_t, const Allocator& a, const indirect& other);

#include <cassert>
#include <memory>

#include "test_allocator.h"
#include "test_convertible.h"

constexpr void test_copy_ctor_not_explicit() {
  static_assert(test_convertible<std::indirect<int>, const std::indirect<int>&>());
  static_assert(test_convertible<std::indirect<int>,
                                 std::allocator_arg_t,
                                 const std::allocator<int>&,
                                 const std::indirect<int>&>());
}

constexpr void test_copy_ctor() {
  {
    const std::indirect<int> i1(42);
    std::indirect<int> i2(i1);
    assert(!i1.valueless_after_move());
    assert(!i2.valueless_after_move());
    assert(*i1 == 42);
    assert(*i2 == 42);
    assert(&*i1 != &*i2);
  }
  {
    std::indirect<int> i1;
    auto(std::move(i1));
    assert(i1.valueless_after_move());
    std::indirect<int> i2(i1);
    assert(i2.valueless_after_move());
  }
  {
    std::indirect<int, SocccAllocator<int>> i1;
    assert(i1.get_allocator().count_ == 0);
    std::indirect<int, SocccAllocator<int>> i2(i1);
    assert(i2.get_allocator().count_ == 1);
  }
  {
    const std::indirect<int> i1(42);
    std::indirect<int> i2(std::allocator_arg, std::allocator<int>(), i1);
    assert(!i1.valueless_after_move());
    assert(!i2.valueless_after_move());
    assert(*i1 == 42);
    assert(*i2 == 42);
    assert(&*i1 != &*i2);
  }
  {
    const std::indirect<int, test_allocator<int>> i1;
    std::indirect<int, test_allocator<int>> i2(std::allocator_arg, test_allocator<int>(42), i1);
    assert(i1.get_allocator().get_data() == 0);
    assert(i2.get_allocator().get_data() == 42);
  }
}

void test_copy_ctor_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct CopyCtorThrows {
    CopyCtorThrows() = default;
    CopyCtorThrows(const CopyCtorThrows&) { throw 42; }
  };

  {
    const std::indirect<CopyCtorThrows> i1;
    try {
      std::indirect<CopyCtorThrows> i2(i1);
      assert(false);
    } catch (const int& e) {
      assert(e == 42);
    } catch (...) {
      assert(false);
    }
  }
  {
    const std::indirect<CopyCtorThrows> i1;
    try {
      std::indirect<CopyCtorThrows> i2(std::allocator_arg, std::allocator<int>(), i1);
      assert(false);
    } catch (const int& e) {
      assert(e == 42);
    } catch (...) {
      assert(false);
    }
  }
#endif
}

constexpr bool test() {
  test_copy_ctor_not_explicit();
  test_copy_ctor();

  return true;
}

int main(int, char**) {
  test_copy_ctor_throws();
  test();
  static_assert(test());
  return 0;
}
