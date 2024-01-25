//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Regression test to error in deque::__annotate_from_to in deque,
// with origin in deque::__add_back_capacity.

// `check_assertion.h` is only available starting from C++11 and requires Unix headers and regex support.
// UNSUPPORTED: c++03, !has-unix-headers, no-localization

#include <deque>
#include <cstdio>
#include "check_assertion.h"

void test1() {
  std::deque<char> test;
  char buff[100000];
  test.insert(test.begin(), buff, buff + 64000);

  for (int i = 0; i < 1100; i += 1) {
    test.insert(test.begin(), buff, buff + 320);
    test.erase(test.end() - 320, test.end());
  }

  test.insert(test.begin(), buff, buff + 32000);
}

void test2() {
  std::deque<char> test;
  char buff[100000];
  test.insert(test.end(), buff, buff + 64000);

  for (int i = 0; i < 1100; i += 1) {
    test.insert(test.end(), buff, buff + 320);
    test.erase(test.begin(), test.begin() + 320);
  }

  test.insert(test.end(), buff, buff + 32000);
}

int main(int, char**) {
  test1();
  test2();

  return 0;
}
