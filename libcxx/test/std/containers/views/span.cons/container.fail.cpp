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
#include <deque>
#include <forward_list>
#include <list>
#include <vector>

#include "test_macros.h"

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer {
    constexpr IsAContainer() : v_{} {}
    constexpr size_t size() const {return 1;}
    constexpr       T *data() {return &v_;}
    constexpr const T *data() const {return &v_;}

    constexpr const T *getV() const {return &v_;} // for checking
    T v_;
};

template <typename T>
struct NotAContainerNoData {
    size_t size() const {return 0;}
};

template <typename T>
struct NotAContainerNoSize {
    const T *data() const {return nullptr;}
};

template <typename T>
struct NotAContainerPrivate {
private:
    size_t size() const {return 0;}
    const T *data() const {return nullptr;}
};


int main(int, char**)
{

//  Making non-const spans from const sources (a temporary binds to `const &`)
    {
    std::span<int>    s1{IsAContainer<int>()};          // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<int, 0> s2{IsAContainer<int>()};          // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}} 
    std::span<int>    s3{std::vector<int>()};           // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<int, 0> s4{std::vector<int>()};           // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}} 
    }

//  Missing size and/or data
    {
    std::span<const int>    s1{NotAContainerNoData<int>()};   // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const int, 0> s2{NotAContainerNoData<int>()};   // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    std::span<const int>    s3{NotAContainerNoSize<int>()};   // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const int, 0> s4{NotAContainerNoSize<int>()};   // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    std::span<const int>    s5{NotAContainerPrivate<int>()};  // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const int, 0> s6{NotAContainerPrivate<int>()};  // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}

//  Again with the standard containers
    std::span<const int>    s11{std::deque<int>()};           // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const int, 0> s12{std::deque<int>()};           // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    std::span<const int>    s13{std::list<int>()};            // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const int, 0> s14{std::list<int>()};            // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    std::span<const int>    s15{std::forward_list<int>()};    // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const int, 0> s16{std::forward_list<int>()};    // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    }

//  Not the same type
    {
    IsAContainer<int> c;
    std::span<float>    s1{c};   // expected-error {{no matching constructor for initialization of 'std::span<float>'}}
    std::span<float, 0> s2{c};   // expected-error {{no matching constructor for initialization of 'std::span<float, 0>'}}
    }

//  CV wrong (dynamically sized)
    {
    IsAContainer<const          int> c;
    IsAContainer<const volatile int> cv;
    IsAContainer<      volatile int> v;
    
    std::span<               int> s1{c};    // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s2{v};    // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s3{cv};   // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<const          int> s4{v};    // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const          int> s5{cv};   // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<      volatile int> s6{c};    // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    std::span<      volatile int> s7{cv};   // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    }

//  CV wrong (statically sized)
    {
    IsAContainer<const          int> c;
    IsAContainer<const volatile int> cv;
    IsAContainer<      volatile int> v;

    std::span<               int,1> s1{c};  // expected-error {{no matching constructor for initialization of 'std::span<int, 1>'}}
    std::span<               int,1> s2{v};  // expected-error {{no matching constructor for initialization of 'std::span<int, 1>'}}
    std::span<               int,1> s3{cv}; // expected-error {{no matching constructor for initialization of 'std::span<int, 1>'}}
    std::span<const          int,1> s4{v};  // expected-error {{no matching constructor for initialization of 'std::span<const int, 1>'}}
    std::span<const          int,1> s5{cv}; // expected-error {{no matching constructor for initialization of 'std::span<const int, 1>'}}
    std::span<      volatile int,1> s6{c};  // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 1>'}}
    std::span<      volatile int,1> s7{cv}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 1>'}}
    }


  return 0;
}
