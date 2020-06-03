//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

// UNSUPPORTED: c++03

#include <tuple>
#include <cassert>
#include <type_traits>
#include <string>
#include <system_error>

#include "test_macros.h"
#include "test_convertible.h"
#include "MoveOnly.h"

#if defined(_LIBCPP_ENABLE_TUPLE_IMPLICIT_REDUCED_ARITY_EXTENSION)
#error This macro should not be defined by default
#endif

struct NoDefault { NoDefault() = delete; };


// Make sure the _Up... constructor SFINAEs out when the types that
// are not explicitly initialized are not all default constructible.
// Otherwise, std::is_constructible would return true but instantiating
// the constructor would fail.
void test_default_constructible_extension_sfinae()
{
    typedef MoveOnly MO;
    typedef NoDefault ND;
    {
        typedef std::tuple<MO, ND> Tuple;
        static_assert(!std::is_constructible<Tuple, MO>::value, "");
        static_assert(std::is_constructible<Tuple, MO, ND>::value, "");
        static_assert(test_convertible<Tuple, MO, ND>(), "");
    }
    {
        typedef std::tuple<MO, MO, ND> Tuple;
        static_assert(!std::is_constructible<Tuple, MO, MO>::value, "");
        static_assert(std::is_constructible<Tuple, MO, MO, ND>::value, "");
        static_assert(test_convertible<Tuple, MO, MO, ND>(), "");
    }
    {
        // Same idea as above but with a nested tuple type.
        typedef std::tuple<MO, ND> Tuple;
        typedef std::tuple<MO, Tuple, MO, MO> NestedTuple;

        static_assert(!std::is_constructible<
            NestedTuple, MO, MO, MO, MO>::value, "");
        static_assert(std::is_constructible<
            NestedTuple, MO, Tuple, MO, MO>::value, "");
    }
}

using ExplicitTup = std::tuple<std::string, int, std::error_code>;
ExplicitTup doc_example() {
      return ExplicitTup{"hello world", 42}; // explicit constructor called. OK.
}

// Test that the example given in UsingLibcxx.rst actually works.
void test_example_from_docs() {
  auto tup = doc_example();
  assert(std::get<0>(tup) == "hello world");
  assert(std::get<1>(tup) == 42);
  assert(std::get<2>(tup) == std::error_code{});
}

int main(int, char**)
{
    {
        using E = MoveOnly;
        using Tup = std::tuple<E, E, E>;
        // Test that the reduced arity initialization extension is only
        // allowed on the explicit constructor.
        static_assert(test_convertible<Tup, E, E, E>(), "");

        Tup t(E(0), E(1));
        static_assert(std::is_constructible<Tup, E, E>::value, "");
        static_assert(!test_convertible<Tup, E, E>(), "");
        assert(std::get<0>(t) == E(0));
        assert(std::get<1>(t) == E(1));
        assert(std::get<2>(t) == E());

        Tup t2(E(0));
        static_assert(std::is_constructible<Tup, E>::value, "");
        static_assert(!test_convertible<Tup, E>(), "");
        assert(std::get<0>(t2) == E(0));
        assert(std::get<1>(t2) == E());
        assert(std::get<2>(t2) == E());
    }
    // Check that SFINAE is properly applied with the default reduced arity
    // constructor extensions.
    test_default_constructible_extension_sfinae();
    test_example_from_docs();

  return 0;
}
