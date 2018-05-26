//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides


// template<class T>
//   optional(T) -> optional<T>;


#include <optional>
#include <cassert>

struct A {};

int main()
{  
//  Test the explicit deduction guides

//  Test the implicit deduction guides
    {
//  optional()
    std::optional opt;   // expected-error {{declaration of variable 'opt' with deduced type 'std::optional' requires an initializer}}
    }

    {
//  optional(nullopt_t)
    std::optional opt(std::nullopt);   // expected-error-re@optional:* {{static_assert failed{{.*}} "instantiation of optional with nullopt_t is ill-formed"}}
    }
}
