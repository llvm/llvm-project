//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template<size_t N>
//     constexpr span(element_type (&arr)[N]) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   - extent == dynamic_extent || N == extent is true, and
//   - remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//

#include <cassert>
#include <span>
#include <string>
#include <type_traits>

#include "test_macros.h"


void checkCV()
{
                   int   arr[] = {1,2,3};
    const          int  carr[] = {4,5,6};
          volatile int  varr[] = {7,8,9};
    const volatile int cvarr[] = {1,3,5};

//  Types the same (dynamic sized)
    {
    std::span<               int> s1{  arr};    // a span<               int> pointing at int.
    std::span<const          int> s2{ carr};    // a span<const          int> pointing at const int.
    std::span<      volatile int> s3{ varr};    // a span<      volatile int> pointing at volatile int.
    std::span<const volatile int> s4{cvarr};    // a span<const volatile int> pointing at const volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 12);
    }

//  Types the same (static sized)
    {
    std::span<               int,3> s1{  arr};  // a span<               int> pointing at int.
    std::span<const          int,3> s2{ carr};  // a span<const          int> pointing at const int.
    std::span<      volatile int,3> s3{ varr};  // a span<      volatile int> pointing at volatile int.
    std::span<const volatile int,3> s4{cvarr};  // a span<const volatile int> pointing at const volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 12);
    }


//  types different (dynamic sized)
    {
    std::span<const          int> s1{ arr};     // a span<const          int> pointing at int.
    std::span<      volatile int> s2{ arr};     // a span<      volatile int> pointing at int.
    std::span<      volatile int> s3{ arr};     // a span<      volatile int> pointing at const int.
    std::span<const volatile int> s4{ arr};     // a span<const volatile int> pointing at int.
    std::span<const volatile int> s5{carr};     // a span<const volatile int> pointing at const int.
    std::span<const volatile int> s6{varr};     // a span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() + s5.size() + s6.size() == 18);
    }

//  types different (static sized)
    {
    std::span<const          int,3> s1{ arr};   // a span<const          int> pointing at int.
    std::span<      volatile int,3> s2{ arr};   // a span<      volatile int> pointing at int.
    std::span<      volatile int,3> s3{ arr};   // a span<      volatile int> pointing at const int.
    std::span<const volatile int,3> s4{ arr};   // a span<const volatile int> pointing at int.
    std::span<const volatile int,3> s5{carr};   // a span<const volatile int> pointing at const int.
    std::span<const volatile int,3> s6{varr};   // a span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() + s5.size() + s6.size() == 18);
    }
}

template<class T>
constexpr bool testSpan()
{
    T val[2] = {};

    ASSERT_NOEXCEPT(std::span<T>{val});
    ASSERT_NOEXCEPT(std::span<T, 2>{val});
    ASSERT_NOEXCEPT(std::span<const T>{val});
    ASSERT_NOEXCEPT(std::span<const T, 2>{val});

    std::span<T> s1 = val;
    std::span<T, 2> s2 = val;
    std::span<const T> s3 = val;
    std::span<const T, 2> s4 = val;
    assert(s1.data() == val && s1.size() == 2);
    assert(s2.data() == val && s2.size() == 2);
    assert(s3.data() == val && s3.size() == 2);
    assert(s4.data() == val && s4.size() == 2);

    std::span<const int> s5 = {{1,2}};
#if TEST_STD_VER >= 26
    std::span<const int, 2> s6({1, 2});
#else
    std::span<const int, 2> s6 = {{1,2}};
#endif
    assert(s5.size() == 2);  // and it dangles
    assert(s6.size() == 2);  // and it dangles

    return true;
}


struct A {};

int main(int, char**)
{
    testSpan<int>();
    testSpan<double>();
    testSpan<A>();
    testSpan<std::string>();

    static_assert(testSpan<int>());
    static_assert(testSpan<double>());
    static_assert(testSpan<A>());

    checkCV();

    // Size wrong
    {
        static_assert(!std::is_constructible<std::span<int, 2>, int (&)[3]>::value, "");
    }

    // Type wrong
    {
        static_assert(!std::is_constructible<std::span<float>,    int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<float, 3>, int (&)[3]>::value, "");
    }

    // CV wrong (dynamically sized)
    {
        static_assert(!std::is_constructible<std::span<int>, const int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<int>, volatile int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<int>, const volatile int (&)[3]>::value, "");

        static_assert(!std::is_constructible<std::span<const int>, volatile int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<const int>, const volatile int (&)[3]>::value, "");

        static_assert(!std::is_constructible<std::span<volatile int>, const int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<volatile int>, const volatile int (&)[3]>::value, "");
    }

    // CV wrong (statically sized)
    {
        static_assert(!std::is_constructible<std::span<int, 3>, const int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<int, 3>, volatile int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<int, 3>, const volatile int (&)[3]>::value, "");

        static_assert(!std::is_constructible<std::span<const int, 3>, volatile int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<const int, 3>, const volatile int (&)[3]>::value, "");

        static_assert(!std::is_constructible<std::span<volatile int, 3>, const int (&)[3]>::value, "");
        static_assert(!std::is_constructible<std::span<volatile int, 3>, const volatile int (&)[3]>::value, "");
    }

    return 0;
}
