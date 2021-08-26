//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// template <class T1, class T2> struct pair

// explicit(see-below) constexpr pair();

// This test checks the conditional explicitness of std::pair's default
// constructor as introduced by the resolution of LWG 2510.

#include <utility>


struct ImplicitlyDefaultConstructible {
    ImplicitlyDefaultConstructible() = default;
};

struct ExplicitlyDefaultConstructible {
    explicit ExplicitlyDefaultConstructible() = default;
};

std::pair<ImplicitlyDefaultConstructible, ExplicitlyDefaultConstructible> test1() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::pair<ExplicitlyDefaultConstructible, ImplicitlyDefaultConstructible> test2() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::pair<ExplicitlyDefaultConstructible, ExplicitlyDefaultConstructible> test3() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::pair<ImplicitlyDefaultConstructible, ImplicitlyDefaultConstructible> test4() { return {}; }

int main(int, char**) {
    return 0;
}
