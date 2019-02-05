//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// float_denorm_style

#include <limits>

typedef char one;
struct two {one _[2];};

one test(std::float_denorm_style);
two test(int);

int main(int, char**)
{
    static_assert(std::denorm_indeterminate == -1,
                 "std::denorm_indeterminate == -1");
    static_assert(std::denorm_absent == 0,
                 "std::denorm_absent == 0");
    static_assert(std::denorm_present == 1,
                 "std::denorm_present == 1");
    static_assert(sizeof(test(std::denorm_present)) == 1,
                 "sizeof(test(std::denorm_present)) == 1");
    static_assert(sizeof(test(1)) == 2,
                 "sizeof(test(1)) == 2");

  return 0;
}
