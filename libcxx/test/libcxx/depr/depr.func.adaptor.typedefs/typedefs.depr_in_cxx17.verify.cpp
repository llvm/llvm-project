//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include "test_macros.h"

void test_functional()
{
    {
        using T = std::plus<int>;
        T::result_type a;          // expected-warning {{is deprecated}}
        T::first_argument_type b;  // expected-warning {{is deprecated}}
        T::second_argument_type c; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
        (void)c;
    }
    {
        using T = std::less<int>;
        T::result_type a;          // expected-warning {{is deprecated}}
        T::first_argument_type b;  // expected-warning {{is deprecated}}
        T::second_argument_type c; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
        (void)c;
    }
    {
        using T = std::logical_not<int>;
        T::result_type a;    // expected-warning {{is deprecated}}
        T::argument_type b;  // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
    }
}

void test_owner_less()
{
    {
        using T = std::owner_less<std::shared_ptr<int>>;
        T::result_type a;          // expected-warning {{is deprecated}}
        T::first_argument_type b;  // expected-warning {{is deprecated}}
        T::second_argument_type c; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
        (void)c;
    }
    {
        using T = std::owner_less<std::weak_ptr<int>>;
        T::result_type a;          // expected-warning {{is deprecated}}
        T::first_argument_type b;  // expected-warning {{is deprecated}}
        T::second_argument_type c; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
        (void)c;
    }
}

void test_hash()
{
    {
        using T = std::hash<int>;
        T::result_type a;   // expected-warning {{is deprecated}}
        T::argument_type b; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
    }
    {
        using T = std::hash<std::shared_ptr<int>>;
        T::result_type a;   // expected-warning {{is deprecated}}
        T::argument_type b; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
    }
    {
        using T = std::hash<std::unique_ptr<int>>;
        T::result_type a;   // expected-warning {{is deprecated}}
        T::argument_type b; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
    }
    {
        using T = std::hash<std::optional<int>>;
        T::result_type a;   // expected-warning {{is deprecated}}
        T::argument_type b; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
    }
}

void test_map()
{
    {
        using T = std::map<int, int>::value_compare;
        T::result_type a;          // expected-warning {{is deprecated}}
        T::first_argument_type b;  // expected-warning {{is deprecated}}
        T::second_argument_type c; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
        (void)c;
    }
    {
        using T = std::multimap<int, int>::value_compare;
        T::result_type a;          // expected-warning {{is deprecated}}
        T::first_argument_type b;  // expected-warning {{is deprecated}}
        T::second_argument_type c; // expected-warning {{is deprecated}}
        (void)a;
        (void)b;
        (void)c;
    }
}
