//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// __resize_default_init(size_type)

#include <string>
#include <cassert>

#include "test_macros.h"

void write_c_str(char *buf, int size) {
  for (int i=0; i < size; ++i) {
    buf[i] = 'a';
  }
  buf[size] = '\0';
}

void test_buffer_usage()
{
  {
    unsigned buff_size = 125;
    unsigned used_size = buff_size - 16;
    std::string s;
    s.__resize_default_init(buff_size);
    write_c_str(&s[0], used_size);
    assert(s.size() == buff_size);
    assert(strlen(s.data()) == used_size);
    s.__resize_default_init(used_size);
    assert(s.size() == used_size);
    assert(s.data()[used_size] == '\0');
    for (unsigned i=0; i < used_size; ++i) {
      assert(s[i] == 'a');
    }
  }
}

void test_basic() {
  {
    std::string s;
    s.__resize_default_init(3);
    assert(s.size() == 3);
    assert(s.data()[3] == '\0');
    for (int i=0; i < 3; ++i)
      s[i] = 'a' + i;
    s.__resize_default_init(1);
    assert(s[0] == 'a');
    assert(s.data()[1] == '\0');
    assert(s.size() == 1);
  }
}

int main(int, char**) {
  test_basic();
  test_buffer_usage();

  return 0;
}
