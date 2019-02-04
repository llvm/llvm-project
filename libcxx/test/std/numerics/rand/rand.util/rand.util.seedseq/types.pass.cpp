//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class seed_seq
// {
// public:
//     // types
//     typedef uint_least32_t result_type;

#include <random>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::seed_seq::result_type, std::uint_least32_t>::value), "");
}
