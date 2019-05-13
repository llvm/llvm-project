// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// LWG issue 3024

#include <variant>
#include <type_traits>

struct NotCopyConstructible
{
    NotCopyConstructible() = default;
    NotCopyConstructible(NotCopyConstructible const&) = delete;
};

int main(int, char**)
{
    static_assert(!std::is_copy_constructible_v<NotCopyConstructible>);

    std::variant<NotCopyConstructible> v;
    std::variant<NotCopyConstructible> v1;
    std::variant<NotCopyConstructible> v2(v); // expected-error {{call to implicitly-deleted copy constructor of 'std::variant<NotCopyConstructible>'}}
    v1 = v; // expected-error {{object of type 'std::__1::variant<NotCopyConstructible>' cannot be assigned because its copy assignment operator is implicitly deleted}}
}
