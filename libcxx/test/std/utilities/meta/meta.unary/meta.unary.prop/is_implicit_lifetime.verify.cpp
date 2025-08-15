//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// These compilers don't support __builtin_is_implicit_lifetime yet.
// UNSUPPORTED: clang-19, gcc-14, apple-clang-16, apple-clang-17

// <type_traits>

// template<class T> struct is_implicit_lifetime;

#include <type_traits>

struct IncompleteStruct;

// expected-error@*:* {{incomplete type 'IncompleteStruct' used in type trait expression}}
static_assert(!std::is_implicit_lifetime<IncompleteStruct>::value);

// expected-error@*:* {{incomplete type 'IncompleteStruct' used in type trait expression}}
static_assert(!std::is_implicit_lifetime_v<IncompleteStruct>);
