//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <optional>

// template <class T> constexpr bool operator==(const optional<T>& x, const optional<T>& y);

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;

struct X
{
    int i_;

    constexpr X(int i) : i_(i) {}
};

constexpr bool operator == ( const X &lhs, const X &rhs )
    { return lhs.i_ == rhs.i_ ; }

int main()
{
    {
    typedef X T;
    typedef optional<T> O;

    constexpr O o1;     // disengaged
    constexpr O o2;     // disengaged
    constexpr O o3{1};  // engaged
    constexpr O o4{2};  // engaged
    constexpr O o5{1};  // engaged

    static_assert (   o1 == o1 , "" );
    static_assert (   o1 == o2 , "" );
    static_assert ( !(o1 == o3), "" );
    static_assert ( !(o1 == o4), "" );
    static_assert ( !(o1 == o5), "" );

    static_assert (   o2 == o1 , "" );
    static_assert (   o2 == o2 , "" );
    static_assert ( !(o2 == o3), "" );
    static_assert ( !(o2 == o4), "" );
    static_assert ( !(o2 == o5), "" );

    static_assert ( !(o3 == o1), "" );
    static_assert ( !(o3 == o2), "" );
    static_assert (   o3 == o3 , "" );
    static_assert ( !(o3 == o4), "" );
    static_assert (   o3 == o5 , "" );

    static_assert ( !(o4 == o1), "" );
    static_assert ( !(o4 == o2), "" );
    static_assert ( !(o4 == o3), "" );
    static_assert (   o4 == o4 , "" );
    static_assert ( !(o4 == o5), "" );

    static_assert ( !(o5 == o1), "" );
    static_assert ( !(o5 == o2), "" );
    static_assert (   o5 == o3 , "" );
    static_assert ( !(o5 == o4), "" );
    static_assert (   o5 == o5 , "" );

    }
}
