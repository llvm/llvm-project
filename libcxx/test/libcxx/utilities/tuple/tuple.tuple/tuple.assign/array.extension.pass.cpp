//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// EXTENSION
// template <class U, size_t N>
//   tuple& operator=(const array<U, N>& u);
//
// template <class U, size_t N>
//   tuple& operator=(array<U, N>&& u);

// UNSUPPORTED: c++03

#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>


template <class T>
struct NothrowAssignableFrom {
    NothrowAssignableFrom& operator=(T) noexcept { return *this; }
};

template <class T>
struct PotentiallyThrowingAssignableFrom {
    PotentiallyThrowingAssignableFrom& operator=(T) { return *this; }
};

int main(int, char**) {
    // Tests for the array const& overload
    {
        std::array<long, 3> array = {1l, 2l, 3l};
        std::tuple<int, int, int> tuple;
        tuple = array;
        assert(std::get<0>(tuple) == 1);
        assert(std::get<1>(tuple) == 2);
        assert(std::get<2>(tuple) == 3);
    }
    {
        typedef std::tuple<NothrowAssignableFrom<int>> Tuple;
        typedef std::array<int, 1> Array;
        static_assert(std::is_nothrow_assignable<Tuple&, Array const&>::value, "");
    }
    {
        typedef std::tuple<PotentiallyThrowingAssignableFrom<int>> Tuple;
        typedef std::array<int, 1> Array;
        static_assert(std::is_assignable<Tuple&, Array const&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Array const&>::value, "");
    }

    // Tests for the array&& overload
    {
        std::array<long, 3> array = {1l, 2l, 3l};
        std::tuple<int, int, int> tuple;
        tuple = std::move(array);
        assert(std::get<0>(tuple) == 1);
        assert(std::get<1>(tuple) == 2);
        assert(std::get<2>(tuple) == 3);
    }
    {
        typedef std::tuple<NothrowAssignableFrom<int>> Tuple;
        typedef std::array<int, 1> Array;
        static_assert(std::is_nothrow_assignable<Tuple&, Array&&>::value, "");
    }
    {
        typedef std::tuple<PotentiallyThrowingAssignableFrom<int>> Tuple;
        typedef std::array<int, 1> Array;
        static_assert(std::is_assignable<Tuple&, Array&&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Array&&>::value, "");
    }

    // Test lvalue-refs and const rvalue-ref
    {
        typedef std::tuple<NothrowAssignableFrom<int>> Tuple;
        typedef std::array<int, 1> Array;
        static_assert(std::is_nothrow_assignable<Tuple&, Array&>::value, "");
        static_assert(std::is_nothrow_assignable<Tuple&, const Array&&>::value, "");
    }

    {
        typedef std::tuple<NothrowAssignableFrom<int>> Tuple;
        static_assert(!std::is_assignable<Tuple&, std::array<long, 2>&>::value, "");
        static_assert(!std::is_assignable<Tuple&, std::array<long, 2>&&>::value, "");
        static_assert(!std::is_assignable<Tuple&, const std::array<long, 2>&>::value, "");
        static_assert(!std::is_assignable<Tuple&, const std::array<long, 2>&&>::value, "");

        static_assert(!std::is_assignable<Tuple&, std::array<long, 4>&>::value, "");
        static_assert(!std::is_assignable<Tuple&, std::array<long, 4>&&>::value, "");
        static_assert(!std::is_assignable<Tuple&, const std::array<long, 4>&>::value, "");
        static_assert(!std::is_assignable<Tuple&, const std::array<long, 4>&&>::value, "");
    }

    return 0;
}
