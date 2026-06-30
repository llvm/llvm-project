//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// See https://llvm.org/PR20855

#include <tuple>
#include <string>

#include "test_macros.h"

template <class Tp>
struct ConvertsTo {
  using RawTp = typename std::remove_cv< typename std::remove_reference<Tp>::type>::type;

  operator Tp() const {
    return static_cast<Tp>(value);
  }

  mutable RawTp value;
};

struct Base {};
struct Derived : Base {};

template <class T> struct CannotDeduce {
 using type = T;
};

template <class ...Args>
void F(typename CannotDeduce<std::tuple<Args...>>::type const&) {}

void f() {
  // Constructing a reference element that would bind to a temporary is rejected. Since C++23
  // ([tuple.cnstr]) the offending constructors are deleted, so the error is reported at the call;
  // before C++23 the bind is caught by a static_assert in the library (with Clang additionally
  // diagnosing the dangling reference member).
#if TEST_STD_VER >= 23
  // expected-error@*:* 8 {{deleted}}
#else
  // expected-error@tuple:* 8 {{Attempted construction of reference element binds to a temporary whose lifetime has ended}}
  // expected-error@tuple:* 0+ {{reference member '__value_' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
#endif

  { F<int, const std::string&>(std::make_tuple(1, "abc")); }
  { std::tuple<int, const std::string&> t(1, "a"); }
  { F<int, const std::string&>(std::tuple<int, const std::string&>(1, "abc")); }
  {
    ConvertsTo<int&> ct;
    std::tuple<const long&, int> t(ct, 42);
  }
  {
    ConvertsTo<int> ct;
    std::tuple<int const&, void*> t(ct, nullptr);
  }
  {
    ConvertsTo<Derived> ct;
    std::tuple<Base const&, int> t(ct, 42);
  }
  {
    std::allocator<int> alloc;
    std::tuple<std::string&&> t2("hello");
    std::tuple<std::string&&> t3(std::allocator_arg, alloc, "hello");
  }
}
