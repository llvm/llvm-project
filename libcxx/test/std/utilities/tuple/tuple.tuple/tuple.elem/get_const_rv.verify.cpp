//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   const typename tuple_element<I, tuple<Types...> >::type&&
//   get(const tuple<Types...>&& t);

// UNSUPPORTED: c++03

#include <tuple>

template <class T> void cref(T const&) {}
template <class T> void cref(T const&&) = delete;

std::tuple<int> const tup4() { return std::make_tuple(4); }

void f() {
    // LWG2485: tuple should not open a hole in the type system, get() should
    // imitate [expr.ref]'s rules for accessing data members
    {
        cref(std::get<0>(tup4())); // expected-error {{call to deleted function 'cref'}}
    }
}
