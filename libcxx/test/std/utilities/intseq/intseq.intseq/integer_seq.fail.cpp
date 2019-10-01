//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T... I>
// struct integer_sequence
// {
//     typedef T type;
//
//     static constexpr size_t size() noexcept;
// };

// This test is a conforming extension.  The extension turns undefined behavior
//  into a compile-time error.

#include <utility>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER > 11

//  Should fail to compile, since float is not an integral type
    using floatmix = std::integer_sequence<float>;
    floatmix::value_type I;

#else

X

#endif  // TEST_STD_VER > 11

  return 0;
}
