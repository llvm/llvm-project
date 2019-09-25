//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// Test that we properly provide the trivial copy operations by default.

// FreeBSD provides the old ABI. This test checks the new ABI so we need
// to manually turn it on.
#if defined(__FreeBSD__)
#define _LIBCPP_ABI_UNSTABLE
#endif

#include <utility>
#include <type_traits>
#include <cstdlib>
#include <cstddef>
#include <cassert>

#include "test_macros.h"

#if defined(_LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR)
#error Non-trivial ctor ABI macro defined
#endif

template <class T>
struct HasTrivialABI : std::integral_constant<bool,
    std::is_trivially_destructible<T>::value
    && (!std::is_copy_constructible<T>::value || std::is_trivially_copy_constructible<T>::value)
#if TEST_STD_VER >= 11
   && (!std::is_move_constructible<T>::value || std::is_trivially_move_constructible<T>::value)
#endif
> {};

#if TEST_STD_VER >= 11
struct NonTrivialDtor {
    NonTrivialDtor(NonTrivialDtor const&) = default;
    ~NonTrivialDtor();
};
NonTrivialDtor::~NonTrivialDtor() {}
static_assert(!HasTrivialABI<NonTrivialDtor>::value, "");

struct NonTrivialCopy {
    NonTrivialCopy(NonTrivialCopy const&);
};
NonTrivialCopy::NonTrivialCopy(NonTrivialCopy const&) {}
static_assert(!HasTrivialABI<NonTrivialCopy>::value, "");

struct NonTrivialMove {
    NonTrivialMove(NonTrivialMove const&) = default;
    NonTrivialMove(NonTrivialMove&&);
};
NonTrivialMove::NonTrivialMove(NonTrivialMove&&) {}
static_assert(!HasTrivialABI<NonTrivialMove>::value, "");

struct DeletedCopy {
    DeletedCopy(DeletedCopy const&) = delete;
    DeletedCopy(DeletedCopy&&) = default;
};
static_assert(HasTrivialABI<DeletedCopy>::value, "");

struct TrivialMove {
  TrivialMove(TrivialMove &&) = default;
};
static_assert(HasTrivialABI<TrivialMove>::value, "");

struct Trivial {
    Trivial(Trivial const&) = default;
};
static_assert(HasTrivialABI<Trivial>::value, "");
#endif


void test_trivial()
{
    {
        typedef std::pair<int, short> P;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(HasTrivialABI<P>::value, "");
    }
#if TEST_STD_VER >= 11
    {
        typedef std::pair<int, short> P;
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<NonTrivialDtor, int>;
        static_assert(!std::is_trivially_destructible<P>::value, "");
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(!HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<NonTrivialCopy, int>;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(!HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<NonTrivialMove, int>;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
        static_assert(!HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<DeletedCopy, int>;
        static_assert(!std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<Trivial, int>;
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasTrivialABI<P>::value, "");
    }
    {
        using P = std::pair<TrivialMove, int>;
        static_assert(!std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(std::is_trivially_move_constructible<P>::value, "");
        static_assert(HasTrivialABI<P>::value, "");
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
