//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template<class F>
// function(F) -> function<see-below>;

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides

// The deduction guides for std::function do not handle rvalue-ref qualified
// call operators and C-style variadics. It also doesn't deduce from nullptr_t.
// Make sure we stick to the specification.

#include <functional>
#include <type_traits>


struct R { };
struct f0 { R operator()() && { return {}; } };
struct f1 { R operator()(int, ...) { return {}; } };

int main(int, char**) {
    std::function f = f0{}; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    std::function g = f1{}; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    std::function h = nullptr; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'function'}}
    return 0;
}
