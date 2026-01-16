//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: gcc-15, apple-clang-17

// <type_traits>

// template <class T>
//   consteval bool is_within_lifetime(const T*) noexcept; // C++26

#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<int*>())), bool);
ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<const int*>())), bool);
ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<void*>())), bool);
ASSERT_SAME_TYPE(decltype(std::is_within_lifetime(std::declval<const void*>())), bool);

ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<int*>()));
ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<const int*>()));
ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<void*>()));
ASSERT_NOEXCEPT(std::is_within_lifetime(std::declval<const void*>()));

template <class T>
concept is_within_lifetime_exists = requires(T t) { std::is_within_lifetime(t); };

struct S {};

static_assert(is_within_lifetime_exists<int*>);
static_assert(is_within_lifetime_exists<const int*>);
static_assert(is_within_lifetime_exists<void*>);
static_assert(is_within_lifetime_exists<const void*>);
static_assert(!is_within_lifetime_exists<int>);               // Not a pointer
static_assert(!is_within_lifetime_exists<decltype(nullptr)>); // Not a pointer
static_assert(!is_within_lifetime_exists<void() const>);      // Not a pointer
static_assert(!is_within_lifetime_exists<int S::*>);          // Doesn't accept pointer-to-data-member
static_assert(!is_within_lifetime_exists<void (S::*)()>);     // Doesn't accept pointer-to-member-function
static_assert(!is_within_lifetime_exists<void (*)()>);        // Doesn't match `const T*`

consteval bool f() {
  // Test that it works with global variables whose lifetime is in a
  // different constant expression
  {
    static constexpr int i = 0;
    static_assert(std::is_within_lifetime(&i));
    // (Even when cast to a different type)
    static_assert(std::is_within_lifetime(const_cast<int*>(&i)));
    static_assert(std::is_within_lifetime(static_cast<const void*>(&i)));
    static_assert(std::is_within_lifetime(static_cast<void*>(const_cast<int*>(&i))));
    static_assert(std::is_within_lifetime<const int>(&i));
    static_assert(std::is_within_lifetime<int>(const_cast<int*>(&i)));
    static_assert(std::is_within_lifetime<const void>(static_cast<const void*>(&i)));
    static_assert(std::is_within_lifetime<void>(static_cast<void*>(const_cast<int*>(&i))));
  }

  {
    static constexpr union {
      int member1;
      int member2;
    } u{.member2 = 1};
    static_assert(!std::is_within_lifetime(&u.member1) && std::is_within_lifetime(&u.member2));
  }

  // Test that it works for varibles inside the same constant expression
  {
    int i = 0;
    assert(std::is_within_lifetime(&i));
    // (Even when cast to a different type)
    assert(std::is_within_lifetime(const_cast<int*>(&i)));
    assert(std::is_within_lifetime(static_cast<const void*>(&i)));
    assert(std::is_within_lifetime(static_cast<void*>(const_cast<int*>(&i))));
    assert(std::is_within_lifetime<const int>(&i));
    assert(std::is_within_lifetime<int>(const_cast<int*>(&i)));
    assert(std::is_within_lifetime<const void>(static_cast<const void*>(&i)));
    assert(std::is_within_lifetime<void>(static_cast<void*>(const_cast<int*>(&i))));
  }
  // Anonymous union
  {
    union {
      int member1;
      int member2;
    };
    assert(!std::is_within_lifetime(&member1) && !std::is_within_lifetime(&member2));
    member1 = 1;
    assert(std::is_within_lifetime(&member1) && !std::is_within_lifetime(&member2));
    member2 = 1;
    assert(!std::is_within_lifetime(&member1) && std::is_within_lifetime(&member2));
  }
  // Variant members
  {
    struct X {
      union {
        int member1;
        int member2;
      };
    } x;
    assert(!std::is_within_lifetime(&x.member1) && !std::is_within_lifetime(&x.member2));
    x.member1 = 1;
    assert(std::is_within_lifetime(&x.member1) && !std::is_within_lifetime(&x.member2));
    x.member2 = 1;
    assert(!std::is_within_lifetime(&x.member1) && std::is_within_lifetime(&x.member2));
  }
  // Unions
  {
    union X {
      int member1;
      int member2;
    } x;
    assert(!std::is_within_lifetime(&x.member1) && !std::is_within_lifetime(&x.member2));
    x.member1 = 1;
    assert(std::is_within_lifetime(&x.member1) && !std::is_within_lifetime(&x.member2));
    x.member2 = 1;
    assert(!std::is_within_lifetime(&x.member1) && std::is_within_lifetime(&x.member2));
  }
  {
    S s; // uninitialised
    assert(std::is_within_lifetime(&s));
  }

  return true;
}
static_assert(f());

// Check that it is a consteval (and consteval-propagating) function
// (i.e., taking the address of below will fail because it will be an immediate function)
template <typename T>
constexpr void does_escalate(T p) {
  std::is_within_lifetime(p);
}
template <typename T, void (*)(T) = &does_escalate<T>>
constexpr bool check_escalated(int) {
  return false;
}
template <typename T>
constexpr bool check_escalated(long) {
  return true;
}
static_assert(check_escalated<int*>(0), "");
static_assert(check_escalated<void*>(0), "");
