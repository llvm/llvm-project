//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// XFAIL: target={{.*}}-windows{{.*}}

#include <fstream>
#include <iostream>
#include <cassert>
#include <vector>

#include "assert_macros.h"
#include "platform_support.h"
#include "test_macros.h"

void test_tellg(std::streamoff total_size) {
  std::vector<char> data(8192);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<char>(i % (1 << 8 * sizeof(char)));
  std::string p = get_temp_file_name();
  {
    std::ofstream ofs;
    ofs.open(p, std::ios::out | std::ios::binary);
    assert(ofs.is_open());
    for (std::streamoff size = 0; size < total_size;) {
      std::size_t n = std::min(static_cast<std::streamoff>(data.size()), total_size - size);
      ofs.write(data.data(), n);
      size += n;
    }
    assert(!ofs.fail());
    ofs.close();
  }
  {
    std::ifstream ifs;
    ifs.open(p, std::ios::binary);
    assert(ifs.is_open());
    std::streamoff in_off = ifs.tellg();
    TEST_REQUIRE(in_off == 0, [&] { test_eprintf("in_off = %ld\n", in_off); });
    ifs.seekg(total_size - 20, std::ios::beg);
    in_off = ifs.tellg();
    TEST_REQUIRE(in_off == total_size - 20, [&] {
      test_eprintf("ref = %zu, in_off = %ld\n", total_size - 20, in_off);
    });
    ifs.seekg(10, std::ios::cur);
    in_off = ifs.tellg();
    TEST_REQUIRE(in_off == total_size - 10, [&] {
      test_eprintf("ref = %zu, in_off = %ld\n", total_size - 10, in_off);
    });
    ifs.seekg(0, std::ios::end);
    in_off = ifs.tellg();
    TEST_REQUIRE(in_off == total_size, [&] { test_eprintf("ref = %zu, in_off = %ld\n", total_size, in_off); });
  }
  std::remove(p.c_str());
}

int main(int, char**) {
  static_assert(sizeof(std::streamoff) > 4);
  test_tellg(0x100000042ULL);
  return 0;
}
