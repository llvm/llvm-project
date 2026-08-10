//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class U1, class U2>
//   tuple& operator=(const pair<U1, U2>& u);

// UNSUPPORTED: c++03

#include <cassert>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

struct NothrowCopyAssignable {
    NothrowCopyAssignable(NothrowCopyAssignable const&) = delete;
    NothrowCopyAssignable& operator=(NothrowCopyAssignable const&) noexcept { return *this; }
};
struct PotentiallyThrowingCopyAssignable {
    PotentiallyThrowingCopyAssignable(PotentiallyThrowingCopyAssignable const&) = delete;
    PotentiallyThrowingCopyAssignable& operator=(PotentiallyThrowingCopyAssignable const&) { return *this; }
};

#include "test_macros.h"

TEST_CONSTEXPR_CXX20
bool test()
{
    {
        typedef std::pair<long, char> T0;
        typedef std::tuple<long long, short> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = t0;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == short('a'));
    }
    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif

    {
        // test that the implicitly generated copy assignment operator
        // is properly deleted
        using T = std::tuple<int, int>;
        using P = std::tuple<std::unique_ptr<int>, std::unique_ptr<int>>;
        static_assert(!std::is_assignable<T&, const P &>::value, "");
    }
    {
        typedef std::tuple<NothrowCopyAssignable, long> Tuple;
        typedef std::pair<NothrowCopyAssignable, int> Pair;
        static_assert(std::is_nothrow_assignable<Tuple&, Pair const&>::value, "");
        static_assert(std::is_nothrow_assignable<Tuple&, Pair&>::value, "");
        static_assert(std::is_nothrow_assignable<Tuple&, Pair const&&>::value, "");
    }
    {
        typedef std::tuple<PotentiallyThrowingCopyAssignable, long> Tuple;
        typedef std::pair<PotentiallyThrowingCopyAssignable, int> Pair;
        static_assert(std::is_assignable<Tuple&, Pair const&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair const&>::value, "");

        static_assert(std::is_assignable<Tuple&, Pair&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair&>::value, "");

        static_assert(std::is_assignable<Tuple&, Pair const&&>::value, "");
        static_assert(!std::is_nothrow_assignable<Tuple&, Pair const&&>::value, "");
    }

    return 0;
}
