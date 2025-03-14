//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// Test that we can seek using offsets larger than 32 bit, and that we can
// retrieve file offsets larger than 32 bit.

// On 32 bit Android platforms, off_t is 32 bit by default. By defining
// _FILE_OFFSET_BITS=64, one gets a 64 bit off_t, but the corresponding
// 64 bit ftello/fseeko functions are only available since Android API 24 (7.0).
// (On 64 bit Android platforms, off_t has always been 64 bit.)
//
// XFAIL: target={{i686|arm.*}}-{{.+}}-android{{.*}}

// Writing the >4 GB test file fails on 32 bit AIX.
//
// XFAIL: target=powerpc-{{.+}}-aix{{.*}}

// By default, off_t is typically a 32-bit integer on ARMv7 Linux systems,
// meaning it can represent file sizes up to 2GB (2^31 bytes) only.
//
// UNSUPPORTED: target=armv7-unknown-linux-gnueabihf

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
    TEST_REQUIRE(in_off == 0, "in_off not zero at start");
    ifs.seekg(total_size - 20, std::ios::beg);
    in_off = ifs.tellg();
    TEST_REQUIRE(in_off == total_size - 20, "in_off incorrect after >32 bit seek");
    ifs.seekg(10, std::ios::cur);
    in_off = ifs.tellg();
    TEST_REQUIRE(in_off == total_size - 10, "in_off incorrect after incremental seek");
    ifs.seekg(0, std::ios::end);
    in_off = ifs.tellg();
    TEST_REQUIRE(in_off == total_size, "in_off incorrect after seek to end");
  }
  std::remove(p.c_str());
}

int main(int, char**) {
  // This test assumes and requires that std::streamoff is larger than
  // 32 bit - this is not required in the standard itself.
  static_assert(sizeof(std::streamoff) > 4, "");
  test_tellg(0x100000042ULL);
  return 0;
}
