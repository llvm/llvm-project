//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

#include <memory>
#include <cassert>

#include "deleter_types.h"
#include "unique_ptr_test_helper.h"

template <int ID = 0>
struct GenericDeleter {
  void operator()(void*) const {}
};

template <int ID = 0>
struct GenericConvertingDeleter {
  template <int OID>
  GenericConvertingDeleter(GenericConvertingDeleter<OID>) {}
  void operator()(void*) const {}
};

void test_sfinae() {
  { // Disallow copying
    using U1 = std::unique_ptr<A[], GenericConvertingDeleter<0> >;
    using U2 = std::unique_ptr<A[], GenericConvertingDeleter<1> >;
    static_assert(std::is_constructible<U1, U2&&>::value, "");
    static_assert(!std::is_constructible<U1, U2&>::value, "");
    static_assert(!std::is_constructible<U1, const U2&>::value, "");
    static_assert(!std::is_constructible<U1, const U2&&>::value, "");
  }
  { // Disallow illegal qualified conversions
    using U1 = std::unique_ptr<const A[]>;
    using U2 = std::unique_ptr<A[]>;
    static_assert(std::is_constructible<U1, U2&&>::value, "");
    static_assert(!std::is_constructible<U2, U1&&>::value, "");
  }
  { // Disallow base-to-derived conversions.
    using UA = std::unique_ptr<A[]>;
    using UB = std::unique_ptr<B[]>;
    static_assert(!std::is_constructible<UA, UB&&>::value, "");
  }
  { // Disallow base-to-derived conversions.
    using UA = std::unique_ptr<A[], GenericConvertingDeleter<0> >;
    using UB = std::unique_ptr<B[], GenericConvertingDeleter<1> >;
    static_assert(!std::is_constructible<UA, UB&&>::value, "");
  }
  { // Disallow invalid deleter initialization
    using U1 = std::unique_ptr<A[], GenericDeleter<0> >;
    using U2 = std::unique_ptr<A[], GenericDeleter<1> >;
    static_assert(!std::is_constructible<U1, U2&&>::value, "");
  }
  { // Disallow reference deleters with different qualifiers
    using U1 = std::unique_ptr<A[], Deleter<A[]>&>;
    using U2 = std::unique_ptr<A[], const Deleter<A[]>&>;
    static_assert(!std::is_constructible<U1, U2&&>::value, "");
    static_assert(!std::is_constructible<U2, U1&&>::value, "");
  }
  {
    using U1 = std::unique_ptr<A[]>;
    using U2 = std::unique_ptr<A>;
    static_assert(!std::is_constructible<U1, U2&&>::value, "");
    static_assert(!std::is_constructible<U2, U1&&>::value, "");
  }
}


int main(int, char**) {
  test_sfinae();

  return 0;
}
