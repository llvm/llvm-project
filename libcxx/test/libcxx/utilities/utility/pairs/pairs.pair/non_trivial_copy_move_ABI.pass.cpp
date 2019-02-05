//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The test fails due to the missing is_trivially_constructible intrinsic.
// XFAIL: gcc-4.9

// The test suite needs to define the ABI macros on the command line when
// modules are enabled.
// UNSUPPORTED: -fmodules

// <utility>

// template <class T1, class T2> struct pair

// Test that we properly provide the trivial copy operations by default.

// FreeBSD provides the old ABI. This test checks the new ABI so we need
// to manually turn it on.
#undef _LIBCPP_ABI_UNSTABLE
#undef _LIBCPP_ABI_VERSION
#define _LIBCPP_ABI_VERSION 1
#define _LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR

#include <utility>
#include <type_traits>
#include <cstdlib>
#include <cstddef>
#include <cassert>

#include "test_macros.h"

#if !defined(_LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR)
#error trivial ctor ABI macro defined
#endif

template <class T>
struct HasNonTrivialABI : std::integral_constant<bool,
    !std::is_trivially_destructible<T>::value
    || (std::is_copy_constructible<T>::value && !std::is_trivially_copy_constructible<T>::value)
#if TEST_STD_VER >= 11
   || (std::is_move_constructible<T>::value && !std::is_trivially_move_constructible<T>::value)
#endif
> {};

#if TEST_STD_VER >= 11
struct NonTrivialDtor {
    NonTrivialDtor(NonTrivialDtor const&) = default;
    ~NonTrivialDtor();
};
NonTrivialDtor::~NonTrivialDtor() {}
static_assert(HasNonTrivialABI<NonTrivialDtor>::value, "");

struct NonTrivialCopy {
    NonTrivialCopy(NonTrivialCopy const&);
};
NonTrivialCopy::NonTrivialCopy(NonTrivialCopy const&) {}
static_assert(HasNonTrivialABI<NonTrivialCopy>::value, "");

struct NonTrivialMove {
    NonTrivialMove(NonTrivialMove const&) = default;
    NonTrivialMove(NonTrivialMove&&);
};
NonTrivialMove::NonTrivialMove(NonTrivialMove&&) {}
static_assert(HasNonTrivialABI<NonTrivialMove>::value, "");

struct DeletedCopy {
    DeletedCopy(DeletedCopy const&) = delete;
    DeletedCopy(DeletedCopy&&) = default;
};
static_assert(!HasNonTrivialABI<DeletedCopy>::value, "");

struct TrivialMove {
  TrivialMove(TrivialMove &&) = default;
};
static_assert(!HasNonTrivialABI<TrivialMove>::value, "");

struct Trivial {
    Trivial(Trivial const&) = default;
};
static_assert(!HasNonTrivialABI<Trivial>::value, "");
#endif


void test_trivial()
{
    {
        typedef std::pair<int, short> P;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
#if TEST_STD_VER >= 11
    {
        typedef std::pair<int, short> P;
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<NonTrivialDtor, int>;
        static_assert(!std::is_trivially_destructible<P>::value, "");
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<NonTrivialCopy, int>;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<NonTrivialMove, int>;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<DeletedCopy, int>;
        static_assert(!std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<Trivial, int>;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<TrivialMove, int>;
        static_assert(!std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasNonTrivialABI<P>::value, "");
    }
#endif
}

void test_layout() {
    typedef std::pair<std::pair<char, char>, char> PairT;
    static_assert(sizeof(PairT) == 3, "");
    static_assert(TEST_ALIGNOF(PairT) == TEST_ALIGNOF(char), "");
    static_assert(offsetof(PairT, first) == 0, "");
}

int main(int, char**) {
    test_trivial();
    test_layout();
    return 0;
}
