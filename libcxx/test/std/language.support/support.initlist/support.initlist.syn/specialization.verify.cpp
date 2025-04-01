//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// template<class E> class initializer_list;
//
// If an explicit specialization or partial specialization of initializer_list
// is declared, the program is ill-formed.

#include <initializer_list>

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else

// expected-error@+2 {{'initializer_list' cannot be specialized: Users are not allowed to specialize this standard library entity}}
template <>
class std::initializer_list<int> {
}; //expected-error 0-1 {{explicit specialization of 'std::initializer_list<int>' after instantiation}}

// expected-error@+2 {{'initializer_list' cannot be specialized: Users are not allowed to specialize this standard library entity}}
template <typename T>
class std::initializer_list<T*> {};

#endif
