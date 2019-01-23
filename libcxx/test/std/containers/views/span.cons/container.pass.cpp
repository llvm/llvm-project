// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

//  template<class Container>
//     constexpr span(Container& cont);
//   template<class Container>
//     constexpr span(const Container& cont);
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — Container is not a specialization of span,
//   — Container is not a specialization of array,
//   — is_array_v<Container> is false,
//   — data(cont) and size(cont) are both well-formed, and
//   — remove_pointer_t<decltype(data(cont))>(*)[] is convertible to ElementType(*)[].
//


#include <span>
#include <cassert>
#include <string>
#include <vector>

#include "test_macros.h"

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer {
    constexpr IsAContainer() : v_{} {}
    constexpr size_t size() const {return 1;}
    constexpr       T *data() {return &v_;}
    constexpr const T *data() const {return &v_;}

    constexpr T const *getV() const {return &v_;} // for checking
    T v_;
};


void checkCV()
{
    std::vector<int> v  = {1,2,3};

//  Types the same (dynamic sized)
    {
    std::span<               int> s1{v};    // a span<               int> pointing at int.
    }

//  Types the same (static sized)
    {
    std::span<               int,3> s1{v};  // a span<               int> pointing at int.
    }

//  types different (dynamic sized)
    {
    std::span<const          int> s1{v};    // a span<const          int> pointing at int.
    std::span<      volatile int> s2{v};    // a span<      volatile int> pointing at int.
    std::span<      volatile int> s3{v};    // a span<      volatile int> pointing at const int.
    std::span<const volatile int> s4{v};    // a span<const volatile int> pointing at int.
    }

//  types different (static sized)
    {
    std::span<const          int,3> s1{v};  // a span<const          int> pointing at int.
    std::span<      volatile int,3> s2{v};  // a span<      volatile int> pointing at int.
    std::span<      volatile int,3> s3{v};  // a span<      volatile int> pointing at const int.
    std::span<const volatile int,3> s4{v};  // a span<const volatile int> pointing at int.
    }

//  Constructing a const view from a temporary
    {
    std::span<const int>    s1{IsAContainer<int>()};
    std::span<const int, 0> s2{IsAContainer<int>()};
    std::span<const int>    s3{std::vector<int>()};
    std::span<const int, 0> s4{std::vector<int>()};
    (void) s1;
    (void) s2;
    (void) s3;
    (void) s4;
    }
}


template <typename T>
constexpr bool testConstexprSpan()
{
    constexpr IsAContainer<const T> val{};
    std::span<const T>    s1{val};
    std::span<const T, 1> s2{val};
    return
        s1.data() == val.getV() && s1.size() == 1
    &&  s2.data() == val.getV() && s2.size() == 1;
}


template <typename T>
void testRuntimeSpan()
{
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    std::span<T>          s1{val};
    std::span<const T>    s2{cVal};
    std::span<T, 1>       s3{val};
    std::span<const T, 1> s4{cVal};
    assert(s1.data() == val.getV()  && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
    assert(s3.data() == val.getV()  && s3.size() == 1);
    assert(s4.data() == cVal.getV() && s4.size() == 1);
}

struct A{};

int main ()
{
    static_assert(testConstexprSpan<int>(),    "");
    static_assert(testConstexprSpan<long>(),   "");
    static_assert(testConstexprSpan<double>(), "");
    static_assert(testConstexprSpan<A>(),      "");

    testRuntimeSpan<int>();
    testRuntimeSpan<long>();
    testRuntimeSpan<double>();
    testRuntimeSpan<std::string>();
    testRuntimeSpan<A>();

    checkCV();
}
