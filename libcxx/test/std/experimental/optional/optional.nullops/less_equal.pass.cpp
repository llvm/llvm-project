//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <optional>

// template <class T> constexpr bool operator<=(const optional<T>& x, nullopt_t) noexcept;
// template <class T> constexpr bool operator<=(nullopt_t, const optional<T>& x) noexcept;

#include <experimental/optional>

#include "test_macros.h"

int main()
{
#if TEST_STD_VER > 11
    using std::experimental::optional;
    using std::experimental::nullopt_t;
    using std::experimental::nullopt;

    {
    typedef int T;
    typedef optional<T> O;

    constexpr O o1;     // disengaged
    constexpr O o2{1};  // engaged

    static_assert (  (nullopt <= o1), "" );
    static_assert (  (nullopt <= o2), "" );
    static_assert (  (o1 <= nullopt), "" );
    static_assert ( !(o2 <= nullopt), "" );

    static_assert (noexcept(nullopt <= o1), "");
    static_assert (noexcept(o1 <= nullopt), "");
    }
#endif
}
