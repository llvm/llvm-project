//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<class F>
// function(F) -> function<see-below>;

// UNSUPPORTED: c++03, c++11, c++14

// The deduction guides for std::function do not handle rvalue-ref qualified
// call operators and C-style variadics. It also doesn't deduce from nullptr_t.
// Make sure we stick to the specification.

#include <functional>
#include <type_traits>

struct R { };
struct f0 { R operator()() && { return {}; } };
struct f1 { R operator()(int, ...) { return {}; } };

void f() {
    std::function f = f0{}; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    std::function g = f1{}; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    std::function h = nullptr; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
}

// LWG 3238. Insufficiently-defined behavior of std::function deduction guides
// https://cplusplus.github.io/LWG/issue3238
template <class T, class = void>
struct IsFunctionDeducible : std::false_type {};

template <class T>
struct IsFunctionDeducible<T, std::void_t<decltype(std::function(std::declval<T>()))>> : std::true_type {};

struct Deducible {
    int operator()() const;
};

static_assert(IsFunctionDeducible<Deducible>::value);
static_assert(!IsFunctionDeducible<f0>::value);
static_assert(!IsFunctionDeducible<f1>::value);
