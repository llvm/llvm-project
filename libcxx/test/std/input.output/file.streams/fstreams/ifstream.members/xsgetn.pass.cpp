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

// This isn't a required override by the standard, but most implementations override it, since it allows for
// significantly improved performance in some cases. All of this code is required to work, so this isn't a libc++
// extension

#include <cassert>
#include <fstream>

#include "test_macros.h"

int main(int, char**) {
  {
    char buffer[10];
    std::ifstream fs("xsgetn.test.dat");
    std::filebuf* fb = fs.rdbuf();
    fb->pubsetbuf(buffer, 10);

    // Ensure that the buffer is set up
    assert(fb->sgetc() == 't');

    std::string str(5, '\0');

    { // Check that a read smaller than the buffer works fine
      assert(fb->sgetn(str.data(), 5) == 5);
      assert(str == "this ");
    }
    { // Check that reading up to the buffer end works fine
      assert(fb->sgetn(str.data(), 5) == 5);
      assert(str == "is so");
    }
    { // Check that reading from an empty buffer, but more than the buffer can hold works fine
      str.resize(12);
      assert(fb->sgetn(str.data(), 12) == 12);
      assert(str == "me random da");
    }
    { // Check that reading from a non-empty buffer, and more than the buffer can hold works fine
      // Fill the buffer up
      str.resize(2);
      assert(fb->sgetn(str.data(), 2) == 2);
      assert(str == "ta");

      // Do the actual check
      str.resize(12);
      assert(fb->sgetn(str.data(), 12) == 12);
      assert(str == " to be able ");
    }
    { // Check that trying to read more than the file size works fine
      str.resize(30);
      assert(fb->sgetn(str.data(), 30) == 24);
      str.resize(24);
      assert(str == "to test buffer behaviour");
    }
  }

  return 0;
}
