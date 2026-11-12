//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// TODO: Revisit this test once we have more information
//       with https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102247
// XFAIL: gcc

// <utility>

// template <class T1, class T2> struct pair
//
// pair(const T1&, const T2&);
// template<class U = T1, class V = T2> pair(U&&, V&&);

// This test checks support for brace-initialization of pairs.

#include <utility>
#include <cassert>

#include "test_macros.h"

struct ExplicitT {
  constexpr explicit ExplicitT(int x) : value(x) {}
  constexpr explicit ExplicitT(ExplicitT const& o) : value(o.value) {}
  int value;
};

struct ImplicitT {
  constexpr ImplicitT(int x) : value(x) {}
  constexpr ImplicitT(ImplicitT const& o) : value(o.value) {}
  int value;
};

template <class T, class = decltype(std::pair<T, T>({}, {}))>
constexpr bool can_construct_with_brace_init(int) { return true; }
template <class T>
constexpr bool can_construct_with_brace_init(...) { return false; }

#if TEST_STD_VER >= 17 // CTAD isn't supported before C++17
template <class T, class = decltype(std::pair(T{}, {}))>
constexpr bool can_construct_with_ctad_brace_init(int) { return true; }
template <class T>
constexpr bool can_construct_with_ctad_brace_init(...) { return false; }
#endif

struct BraceInit { BraceInit() = default; };
struct NoBraceInit { NoBraceInit(int); };
struct ExplicitBraceInit { explicit ExplicitBraceInit() = default; };

constexpr int explicit_vs_implicit_brace_init(std::pair<ExplicitBraceInit, ExplicitBraceInit>) { return 1; }
constexpr int explicit_vs_implicit_brace_init(std::pair<BraceInit, BraceInit>) { return 2; }

TEST_CONSTEXPR_CXX14 bool test() {
    // Explicit constructor
    {
        std::pair<ExplicitT, BraceInit> p1(ExplicitT{42}, {});
        assert(p1.first.value == 42);

        std::pair<ExplicitT, BraceInit> p2{ExplicitT{42}, {}};
        assert(p2.first.value == 42);
    }
    {
        std::pair<BraceInit, ExplicitT> p1({}, ExplicitT{42});
        assert(p1.second.value == 42);

        std::pair<BraceInit, ExplicitT> p2{{}, ExplicitT{42}};
        assert(p2.second.value == 42);
    }
    {
        std::pair<BraceInit, BraceInit> p{{}, {}};
        (void)p;
    }

    // Implicit constructor
    {
        std::pair<ImplicitT, BraceInit> p = {42, {}};
        assert(p.first.value == 42);
    }
    {
        std::pair<BraceInit, ImplicitT> p = {{}, 42};
        assert(p.second.value == 42);
    }
    {
        std::pair<BraceInit, BraceInit> p = {{}, {}};
        (void)p;
    }

    // SFINAE-friendliness of some invalid cases
    {
        static_assert( can_construct_with_brace_init<BraceInit>(0), "");
        static_assert(!can_construct_with_brace_init<NoBraceInit>(0), "");

#if TEST_STD_VER >= 17
        // CTAD with {} should never work, since we can't possibly deduce the types
        static_assert(!can_construct_with_ctad_brace_init<BraceInit>(0), "");
        static_assert(!can_construct_with_ctad_brace_init<int>(0), "");
#endif
    }

    // Make sure there is no ambiguity between the explicit and the non-explicit constructors
    {
        assert(explicit_vs_implicit_brace_init({{}, {}}) == 2);
    }

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 11
    static_assert(test(), "");
#endif

    return 0;
}
