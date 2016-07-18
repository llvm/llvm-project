//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// Doesn't pass due to use of is_trivially_* trait.
// XFAIL: gcc-4.9

// Test that we properly provide the old non-trivial copy operations
// when the ABI macro is defined.

#define _LIBCPP_DEPRECATED_ABI_DISABLE_PAIR_TRIVIAL_COPY_CTOR
#include <utility>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER >= 11
struct Dummy {
  Dummy(Dummy const&) = delete;
  Dummy(Dummy &&) = default;
};
#endif

int main()
{
    typedef std::pair<int, short> P;
    {
        static_assert(std::is_copy_constructible<P>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P>::value, "");
    }
#if TEST_STD_VER >= 11
    {
        static_assert(std::is_move_constructible<P>::value, "");
        static_assert(!std::is_trivially_move_constructible<P>::value, "");
    }
    {
        using P1 = std::pair<Dummy, int>;
        // This line fails because the non-trivial constructors do not provide
        // SFINAE.
        // static_assert(!std::is_copy_constructible<P1>::value, "");
        static_assert(!std::is_trivially_copy_constructible<P1>::value, "");
        static_assert(std::is_move_constructible<P1>::value, "");
        static_assert(!std::is_trivially_move_constructible<P1>::value, "");
    }
#endif
}
