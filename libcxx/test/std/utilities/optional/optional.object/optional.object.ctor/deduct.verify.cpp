//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// template<class T>
//   optional(T) -> optional<T>;

#include <optional>
#include <cassert>

struct A {};

int main(int, char**)
{
//  Test the explicit deduction guides

//  Test the implicit deduction guides
    {
//  optional()
    std::optional opt;   // expected-error-re {{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}optional'}}
    }

    {
//  optional(nullopt_t)
    std::optional opt(std::nullopt);   // expected-error-re@optional:* {{static assertion failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
    }

  return 0;
}
