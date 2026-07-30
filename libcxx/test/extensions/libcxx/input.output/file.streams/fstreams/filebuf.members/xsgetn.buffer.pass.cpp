//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: test.dat

// UNSUPPORTED: no-localization, no-filesystem

// XFAIL: using-built-library-before-llvm-24

// <fstream>

// streamsize basic_filebuf::xsgetn(char_type*, streamsize);

// Test that xsgetn buffers properly. Specifically, we guarantee that `unget()` can be called at least four times.

#include <cassert>
#include <fstream>
#include <string>

#include "platform_support.h"

void small_file_tests() {
  char buffer[12];

  { // Check that we can unget() when reading a single character
    std::ifstream is("test.dat");
    assert(is.is_open());
    char buf[1];
    is.read(buf, 1);
    assert(buf[0] == 'T');
    is.unget();
    assert(is.good());
  }

  { // Check that we can unget() when reading a single character with a user-provided buffer
    std::ifstream is("test.dat");
    assert(is.is_open());
    is.rdbuf()->pubsetbuf(buffer, 12);
    char buf[1];
    is.read(buf, 1);
    assert(buf[0] == 'T');
    is.unget();
    assert(is.good());
  }

  { // Check that unget() works as expected when the remainder is smaller than the buffer
    std::ifstream is("test.dat");
    assert(is.is_open());
    is.rdbuf()->pubsetbuf(buffer, 12);
    (void)is.rdbuf()->sgetc(); // Make sure there is data in the buffer
    char buf[17];
    buf[16] = '\0';
    is.read(buf, 16);
    assert(buf == std::string("This is a bunch "));
    for (size_t i = 0; i != 4; ++i)
      is.unget();
    assert(is.good());
    is.read(buf, 4);
    buf[4] = '\0';
    assert(buf == std::string("nch "));
  }

  { // Check that unget() works as expected when the remainder is larger than the buffer
    std::ifstream is("test.dat");
    assert(is.is_open());
    is.rdbuf()->pubsetbuf(buffer, 12);
    char buf[33];
    buf[32] = '\0';
    is.read(buf, 32);
    assert(buf == std::string("This is a bunch of data so the t"));
    for (size_t i = 0; i != 4; ++i)
      is.unget();
    assert(is.good());
    is.read(buf, 4);
    buf[4] = '\0';
    assert(is.good());
    assert(buf == std::string("he t"));
  }

  { // read an empty file
    std::string empty_file = get_temp_file_name();
    {
      std::ofstream os(empty_file);
    }
    std::ifstream is(empty_file);
    is.rdbuf()->pubsetbuf(nullptr, 64);

    char buf[100];
    is.read(buf, 100);
    assert(is.eof());
    assert(is.gcount() == 0);

    is.clear();
    is.unget();
    assert(is.fail());
    std::remove(empty_file.c_str());
  }
}

static std::string make_pattern(std::size_t n) {
  std::string s(n, '\0');
  for (std::size_t i = 0; i != n; ++i)
    s[i] = static_cast<char>('0' + (i % 10));
  return s;
}

void large_file_tests() {
  const std::string data = make_pattern(10000);
  std::string file       = get_temp_file_name();
  { // Prepare the file
    std::ofstream os(file);
    assert(os.write(data.data(), data.size()));
  }

  { // default buffer with a read larger than the buffer
    std::ifstream is(file);
    assert(is.is_open());
    std::string buf;
    buf.resize(8000);

    is.read(&*buf.begin(), 8000);
    assert(is.gcount() == 8000);
    assert(std::string(buf.data(), 8000) == data.substr(0, 8000));
    is.unget();
    assert(is.good());
    assert(is.get() == data[7999]); // the ungotten character
    assert(is.get() == data[8000]); // reading forward continues from the right place
  }

  { // EOF before the buffer is full
    std::ifstream is(file);
    assert(is.is_open());
    is.rdbuf()->pubsetbuf(nullptr, 64);

    std::string buf;
    buf.resize(data.size() + 100);

    is.read(&*buf.begin(), data.size() + 100);
    assert(is.gcount() == static_cast<std::streamsize>(data.size()));
    assert(is.rdstate() == (std::ios::eofbit | std::ios::failbit));
    is.clear(); // Clear the failbit due to trying to read more data than available
    is.unget();
    assert(is.good());
    assert(is.get() == data.back());
  }

  std::remove(file.c_str());
}

int main(int, char**) {
  small_file_tests();
  large_file_tests();
  return 0;
}
