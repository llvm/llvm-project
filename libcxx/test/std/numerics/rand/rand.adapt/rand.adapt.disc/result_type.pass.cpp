//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t p, size_t r>
// class discard_block_engine
// {
// public:
//     // types
//     typedef typename Engine::result_type result_type;

#include <random>
#include <type_traits>

void
test1()
{
    static_assert((std::is_same<
        std::ranlux24::result_type,
        std::uint_fast32_t>::value), "");
}

void
test2()
{
    static_assert((std::is_same<
        std::ranlux48::result_type,
        std::uint_fast64_t>::value), "");
}

int main(int, char**)
{
    test1();
    test2();

  return 0;
}
