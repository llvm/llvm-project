//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
// class linear_congruential_engine
// {
// public:
//     // types
//     typedef UIntType result_type;

#include <random>
#include <type_traits>

template <class T>
void
test()
{
    static_assert((std::is_same<
        typename std::linear_congruential_engine<T, 0, 0, 0>::result_type,
        T>::value), "");
}

int main(int, char**)
{
    test<unsigned short>();
    test<unsigned int>();
    test<unsigned long>();
    test<unsigned long long>();

  return 0;
}
