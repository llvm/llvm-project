//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_CONST_WRAP_CLASS_HELPERS_H
#define TEST_STD_UTILITIES_CONST_WRAP_CLASS_HELPERS_H

struct NonStructural {
  constexpr NonStructural(int i) : value(i) {}

  constexpr int get() const { return value; }

private:
  int value;
};

#endif // TEST_STD_UTILITIES_CONST_WRAP_CLASS_HELPERS_H
