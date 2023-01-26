//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <memory>
#include <new>

#include "test_macros.h"

typedef std::__compressed_pair<int, unsigned> IntPair;

void test_constructor() {
  IntPair value;
  assert(value.first() == 0);
  assert(value.second() == 0);

  value.first() = 1;
  value.second() = 2;
  new (&value) IntPair;
  assert(value.first() == 0);
  assert(value.second() == 0);
}

void test_constructor_default_init() {
  IntPair value;
  value.first() = 1;
  value.second() = 2;

  new (&value) IntPair(std::__default_init_tag(), 3);
  assert(value.first() == 1);
  assert(value.second() == 3);

  new (&value) IntPair(4, std::__default_init_tag());
  assert(value.first() == 4);
  assert(value.second() == 3);

  new (&value) IntPair(std::__default_init_tag(), std::__default_init_tag());
  assert(value.first() == 4);
  assert(value.second() == 3);
}

int main(int, char**)
{
  test_constructor();
  test_constructor_default_init();
  return 0;
}
