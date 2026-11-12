//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: xsgetn.test.dat

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream

// streamsize xsgetn(char_type*, streamsize) override;

// This isn't a required override by the standard, but most implementations
// override it, since it allows for significantly improved performance in some
// cases. All of this code is required to work, so this isn't a libc++ extension

#include <cassert>
#include <fstream>
#include <vector>

#include "test_macros.h"

void check_unget(std::filebuf* fb, char expected) {
  auto c = fb->sungetc();
  if (c != EOF) {
    assert(c == expected);
    assert(fb->sbumpc() == expected);
  }
}

int main(int, char**) {
  std::vector<char> stream_buffer(10);
  std::ifstream fs("xsgetn.test.dat");
  std::filebuf* fb = fs.rdbuf();
  fb->pubsetbuf(stream_buffer.data(), 10);

  // Ensure that the buffer is set up
  assert(fb->sgetc() == 't');

  std::vector<char> test_buffer(5);
  test_buffer[0] = '\0';

  { // Check that a read smaller than the buffer works fine
    assert(fb->sgetn(test_buffer.data(), 5) == 5);
    assert(std::string(test_buffer.data(), 5) == "this ");
    check_unget(fb, ' ');
  }
  { // Check that reading up to the buffer end works fine
    assert(fb->sgetn(test_buffer.data(), 5) == 5);
    assert(std::string(test_buffer.data(), 5) == "is so");
    check_unget(fb, 'o');
  }
  { // Check that reading from an empty buffer, but more than the buffer can
    // hold works fine
    test_buffer.resize(12);
    assert(fb->sgetn(test_buffer.data(), 12) == 12);
    assert(std::string(test_buffer.data(), 12) == "me random da");
    check_unget(fb, 'a');
  }
  { // Check that reading from a non-empty buffer, and more than the buffer can
    // hold works fine Fill the buffer up
    test_buffer.resize(2);
    assert(fb->sgetn(test_buffer.data(), 2) == 2);
    assert(std::string(test_buffer.data(), 2) == "ta");
    check_unget(fb, 'a');

    // Do the actual check
    test_buffer.resize(12);
    assert(fb->sgetn(test_buffer.data(), 12) == 12);
    assert(std::string(test_buffer.data(), 12) == " to be able ");
    check_unget(fb, ' ');
  }
  { // Check that trying to read more than the file size works fine
    test_buffer.resize(30);
    assert(fb->sgetn(test_buffer.data(), 30) == 24);
    test_buffer.resize(24);
    assert(std::string(test_buffer.data(), 24) == "to test buffer behaviour");
    check_unget(fb, 'r');
  }
  { // Ensure that the read fails gracefully with an unopened ifstream
    // See https://llvm.org/PR168628
    char buf[10];
    std::ifstream ifs;
    std::filebuf* filebuf = ifs.rdbuf();

    assert(!filebuf->is_open());
    assert(filebuf->sgetn(buf, sizeof(buf)) == 0);
    assert(!filebuf->is_open());
  }

  return 0;
}
