//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT
// MSVC warning C4242: '+=': conversion from 'const _Ty' to 'size_t', possible loss of data
// MSVC warning C4244: 'argument': conversion from 'std::streamsize' to 'size_t', possible loss of data
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4242 /wd4244
// UNSUPPORTED: c++03

// <fstream>

// This test checks the behavior of reading payloads of different sizes in different patterns.
// In particular, it was written to exercise code paths that deal with buffering inside the fstream
// implementation.
//
// For each test, we test various behaviors w.r.t. how the buffer is handled:
// - Provide a user-managed buffer to the library. In this case, we test the following corner-cases:
//    + A 0-sized buffer.
//    + A buffer size greater than and smaller than the payload size, which causes multiple buffer effects.
//      Important values are +/- 1 byte from the payload size.
// - Let the library manage a buffer of a user-provided size 'n'. In this case, we test the following corner-cases:
//    + A 0-sized buffer.
//    + A buffer size greater than and smaller than the payload size, which causes multiple buffer effects.
//      Important values are +/- 1 or 2 bytes from the payload size.
//    + A buffer size smaller than 8 bytes. If pubsetbuf() is called with less than 8 bytes, the library will
//      use __extbuf_min_ with 8 bytes instead of allocating anything.
// - Let the library manage a buffer, without specifying any size. In this case, the library will use the default
//   buffer size of 4096 bytes.

#include <cassert>
#include <codecvt>
#include <fstream>
#include <locale>
#include <numeric>
#include <string>
#include <vector>

#include "../types.h"
#include "assert_macros.h"
#include "platform_support.h"
#include "test_macros.h"

template <class BufferPolicy>
void test_read(BufferPolicy policy, const std::vector<std::streamsize>& payload_sizes) {
  std::streamsize total_size = std::accumulate(payload_sizes.begin(), payload_sizes.end(), std::streamsize{0});
  std::vector<char> data(total_size);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>(i % (1 << 8 * sizeof(char)));
  }
  std::string p = get_temp_file_name();
  {
    std::ofstream ofs;
    ofs.open(p, std::ios::out | std::ios::binary);
    assert(ofs.is_open());
    ofs.write(data.data(), data.size());
    assert(!ofs.fail());
    // test that the user's out_buffer buffer was not modified by write()
    for (std::streamsize j = 0; j < total_size; ++j) {
      char exp = j % (1 << 8 * sizeof(char));
      TEST_REQUIRE(data[j] == exp, [&] {
        test_eprintf("failed after write() at offset %zu: got=%x, expected=%x\n", j, data[j], exp);
      });
    }
    ofs.close();
  }
  {
    std::ifstream ifs;
    policy(ifs);
    ifs.open(p, std::ios::ate | std::ios::binary);
    assert(ifs.is_open());
    const std::streamsize in_sz = ifs.tellg();
    TEST_REQUIRE(in_sz == total_size, [&] { test_eprintf("out_sz = %zu, in_sz = %ld\n", total_size, in_sz); });
    ifs.seekg(0, std::ios::beg);
    std::size_t previously_read = 0;
    std::vector<char> in_buffer(total_size);
    for (const auto& payload_sz : payload_sizes) {
      ifs.read(in_buffer.data() + previously_read, payload_sz);
      assert(ifs);
      for (std::streamsize j = 0; j < payload_sz; ++j) {
        char exp = (previously_read + j) % (1 << 8 * sizeof(char));
        TEST_REQUIRE(in_buffer[previously_read + j] == exp, [&] {
          test_eprintf(
              "failed after read() at offset %zu (offset %zu in chunk size %zu): got=%x, expected=%x\n",
              previously_read + j,
              j,
              payload_sz,
              in_buffer[previously_read + j],
              exp);
        });
      }
      previously_read += payload_sz;
    }
  }
  std::remove(p.c_str());
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
template <class BufferPolicy>
void test_read_codecvt(BufferPolicy policy, const std::vector<std::streamsize>& payload_sizes) {
  std::streamsize total_size = std::accumulate(payload_sizes.begin(), payload_sizes.end(), std::streamsize{0});
  std::vector<wchar_t> data(total_size);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<wchar_t>(i);
  }
  std::string p = get_temp_file_name();
  {
    std::wofstream ofs;
    ofs.imbue(std::locale(std::locale::classic(), new std::codecvt_utf8<wchar_t>));
    ofs.open(p, std::ios::out | std::ios::binary);
    assert(ofs.is_open());
    ofs.write(data.data(), data.size());
    assert(!ofs.fail());
    // test that the user's out_buffer buffer was not modified by write()
    for (std::streamsize j = 0; j < total_size; ++j) {
      wchar_t exp = static_cast<wchar_t>(j);
      TEST_REQUIRE(data[j] == exp, [&] {
        test_eprintf("failed after write() at offset %zu: got=%x, expected=%x\n", j, data[j], exp);
      });
    }
    ofs.close();
  }
  {
    std::wifstream ifs;
    ifs.imbue(std::locale(std::locale::classic(), new std::codecvt_utf8<wchar_t>));
    policy(ifs);
    ifs.open(p, std::ios::in | std::ios::binary);
    assert(ifs.is_open());
    ifs.seekg(0, std::ios::beg);
    std::size_t previously_read = 0;
    std::vector<wchar_t> in_buffer(total_size);
    for (const auto& payload_sz : payload_sizes) {
      assert(ifs.read(in_buffer.data() + previously_read, payload_sz));
      assert(ifs);
      for (std::streamsize j = 0; j < payload_sz; ++j) {
        wchar_t exp = static_cast<wchar_t>(j + previously_read);
        TEST_REQUIRE(in_buffer[j + previously_read] == exp, [&] {
          test_eprintf(
              "failed after read() at offset %zu: got=%x, expected=%x\n", j, in_buffer[j + previously_read], exp);
        });
      }
      previously_read += payload_sz;
    }
  }
  std::remove(p.c_str());
}
#endif

const std::vector<std::streamsize> buffer_sizes{0L, 3L, 8L, 9L, 11L};
const std::vector<std::streamsize> io_sizes{0L, 1L, 2L, 3L, 4L, 9L, 10L, 11L, 12L, 13L, 21L, 22L, 23L};
const std::vector<std::streamsize> io_sizes_default{
    0L, 1L, 2L, 3L, 4L, 4094L, 4095L, 4096L, 4097L, 4098L, 8190L, 8191L, 8192L, 8193L, 8194L};

// Test single read operations
void test_1_read() {
  // with default library buffer size: 4096b
  for (std::streamsize x : io_sizes_default) {
    test_read(LibraryDefaultBuffer(), {x});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_read_codecvt(LibraryDefaultBuffer(), {x});
#endif
  }

  // with the library-managed buffer of given size
  for (std::streamsize b : buffer_sizes) {
    for (std::streamsize x : io_sizes) {
      test_read(LibraryManagedBuffer(b), {x});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
      test_read_codecvt(LibraryManagedBuffer(b), {x});
#endif
    }
  }

  // with the user-managed buffer of given size
  for (std::streamsize b : buffer_sizes) {
    for (std::streamsize x : io_sizes) {
      test_read(UserManagedBuffer(b), {x});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
      test_read_codecvt(UserManagedBuffer(b), {x});
#endif
    }
  }
}

// Test two read operations
void test_2_reads() {
  // with default library buffer size: 4096b
  for (std::streamsize a : io_sizes_default) {
    for (std::streamsize b : io_sizes_default) {
      test_read(LibraryDefaultBuffer(), {a, b});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
      test_read_codecvt(LibraryDefaultBuffer(), {a, b});
#endif
    }
  }

  // with the library-managed buffer of given size
  for (std::streamsize buf : buffer_sizes) {
    for (std::streamsize a : io_sizes) {
      for (std::streamsize b : io_sizes) {
        test_read(LibraryManagedBuffer(buf), {a, b});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test_read_codecvt(LibraryManagedBuffer(buf), {a, b});
#endif
      }
    }
  }

  // with the user-managed buffer of given size
  for (std::streamsize buf : buffer_sizes) {
    for (std::streamsize a : io_sizes) {
      for (std::streamsize b : io_sizes) {
        test_read(UserManagedBuffer(buf), {a, b});
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test_read_codecvt(UserManagedBuffer(buf), {a, b});
#endif
      }
    }
  }
}

int main(int, char**) {
  test_1_read();
  test_2_reads();
  return 0;
}
