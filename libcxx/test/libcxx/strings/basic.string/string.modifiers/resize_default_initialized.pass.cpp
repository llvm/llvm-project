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

TEST_CONSTEXPR_CXX20 void write_c_str(char* buf, int size) {
  for (int i = 0; i < size; ++i) {
    buf[i] = 'a';
  }
  buf[size] = '\0';
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_buffer_usage() {
  {
    unsigned buff_size = 125;
    unsigned used_size = buff_size - 16;
    S s;
    s.__resize_default_init(buff_size);
    write_c_str(&s[0], used_size);
    assert(s.size() == buff_size);
    assert(std::char_traits<char>().length(s.data()) == used_size);
    s.__resize_default_init(used_size);
    assert(s.size() == used_size);
    assert(s.data()[used_size] == '\0');
    for (unsigned i = 0; i < used_size; ++i) {
      assert(s[i] == 'a');
    }
  }
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_basic() {
  {
    S s;
    s.__resize_default_init(3);
    assert(s.size() == 3);
    assert(s.data()[3] == '\0');
    for (int i = 0; i < 3; ++i)
      s[i] = 'a' + i;
    s.__resize_default_init(1);
    assert(s[0] == 'a');
    assert(s.data()[1] == '\0');
    assert(s.size() == 1);
  }
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test() {
  test_basic<S>();
  test_buffer_usage<S>();

  return true;
}

int main(int, char**) {
  test<std::string>();
#if TEST_STD_VER > 17
  static_assert(test<std::string>());
#endif

  return 0;
}
