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
  // Test that the public constructors are deleted when constructing a reference
  // element would bind to a temporary.

  {
    F<int, const std::string&>(std::make_tuple(1, "abc")); // expected-error {{deleted}}
  }
  {
    std::tuple<int, const std::string&> t(1, "a"); // expected-error {{deleted}}
  }
  {
    F<int, const std::string&>(std::tuple<int, const std::string&>(1, "abc")); // expected-error {{deleted}}
  }
  {
    ConvertsTo<int&> ct;
    std::tuple<const long&, int> t(ct, 42); // expected-error {{deleted}}
  }
  {
    ConvertsTo<int> ct;
    std::tuple<int const&, void*> t(ct, nullptr); // expected-error {{deleted}}
  }
  {
    ConvertsTo<Derived> ct;
    std::tuple<Base const&, int> t(ct, 42); // expected-error {{deleted}}
  }
  {
    std::allocator<int> alloc;
    std::tuple<std::string&&> t2("hello");                            // expected-error {{deleted}}
    std::tuple<std::string&&> t3(std::allocator_arg, alloc, "hello"); // expected-error {{deleted}}
  }
}
