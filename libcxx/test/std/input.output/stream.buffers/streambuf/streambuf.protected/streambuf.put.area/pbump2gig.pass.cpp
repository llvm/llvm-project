//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <streambuf>

// UNSUPPORTED: c++03

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// void pbump(int n);
//
// REQUIRES: long_tests

// Unsupported for no-exceptions builds because they have no way to report an
// allocation failure when attempting to allocate the 2GiB string.
// UNSUPPORTED: no-exceptions

// Android devices frequently don't have enough memory to run this test. Rather
// than throw std::bad_alloc, exhausting memory tends to trigger the OOM Killer
// and/or crash the device (killing adb, rebooting it, etc).
// UNSUPPORTED: android

#include <sstream>
#include <cassert>
#include "test_macros.h"

struct SB : std::stringbuf
{
  SB() : std::stringbuf(std::ios::ate|std::ios::out) { }
  const char* pubpbase() const { return pbase(); }
  const char* pubpptr() const { return pptr(); }
};

int main(int, char**)
{
    try {
        std::string str(2147483648, 'a');
        SB sb;
        sb.str(str);
        assert(sb.pubpbase() <= sb.pubpptr());
    }
    catch (const std::length_error &) {} // maybe the string can't take 2GB
    catch (const std::bad_alloc    &) {} // maybe we don't have enough RAM

  return 0;
}
