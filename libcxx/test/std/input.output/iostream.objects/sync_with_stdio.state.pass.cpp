//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// Changing the synchronization mode with std::ios_base::sync_with_stdio must
// preserve the per-stream state of the standard stream objects.

#include <cassert>
#include <iostream>

#include "test_macros.h"

int main(int, char**) {
  int dummy       = 0;
  const int index = std::ios_base::xalloc();

  std::cin.iword(index)  = 11;
  std::cout.iword(index) = 22;
  std::cerr.iword(index) = 33;
  std::clog.iword(index) = 44;
  std::cout.pword(index) = &dummy;

  std::cout.setf(std::ios_base::hex, std::ios_base::basefield);
  std::cout.precision(7);
  std::cout.fill('*');

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wcin.iword(index)  = 11;
  std::wcout.iword(index) = 22;
  std::wcerr.iword(index) = 33;
  std::wclog.iword(index) = 44;
  std::wcout.pword(index) = &dummy;

  std::wcout.setf(std::ios_base::hex, std::ios_base::basefield);
  std::wcout.precision(7);
  std::wcout.fill(L'*');
#endif

  // Switch to the unsynchronized mode. This swaps the underlying buffers, but the
  // stream objects and all their state must be preserved.
  std::ios_base::sync_with_stdio(false);

  assert(std::cin.iword(index) == 11);
  assert(std::cout.iword(index) == 22);
  assert(std::cerr.iword(index) == 33);
  assert(std::clog.iword(index) == 44);
  assert(std::cout.pword(index) == &dummy);
  assert((std::cout.flags() & std::ios_base::basefield) == std::ios_base::hex);
  assert(std::cout.precision() == 7);
  assert(std::cout.fill() == '*');
  assert(std::cin.tie() == &std::cout);
  assert(std::cerr.tie() == &std::cout);
  assert(std::cerr.flags() & std::ios_base::unitbuf);
  assert(std::cin.good());
  assert(std::cout.good());
  assert(std::cin.rdbuf());
  assert(std::cout.rdbuf());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert(std::wcin.iword(index) == 11);
  assert(std::wcout.iword(index) == 22);
  assert(std::wcerr.iword(index) == 33);
  assert(std::wclog.iword(index) == 44);
  assert(std::wcout.pword(index) == &dummy);
  assert((std::wcout.flags() & std::ios_base::basefield) == std::ios_base::hex);
  assert(std::wcout.precision() == 7);
  assert(std::wcout.fill() == L'*');
  assert(std::wcin.tie() == &std::wcout);
  assert(std::wcerr.tie() == &std::wcout);
  assert(std::wcerr.flags() & std::ios_base::unitbuf);
  assert(std::wcin.good());
  assert(std::wcout.good());
  assert(std::wcin.rdbuf());
  assert(std::wcout.rdbuf());
#endif

  // Switch back to synchronized mode and check that the state is still there.
  std::ios_base::sync_with_stdio(true);

  assert(std::cout.iword(index) == 22);
  assert(std::cout.pword(index) == &dummy);
  assert((std::cout.flags() & std::ios_base::basefield) == std::ios_base::hex);
  assert(std::cout.precision() == 7);
  assert(std::cout.fill() == '*');
  assert(std::cin.tie() == &std::cout);
  assert(std::cerr.tie() == &std::cout);
  assert(std::cerr.flags() & std::ios_base::unitbuf);
  assert(std::cout.good());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert(std::wcout.iword(index) == 22);
  assert(std::wcout.pword(index) == &dummy);
  assert((std::wcout.flags() & std::ios_base::basefield) == std::ios_base::hex);
  assert(std::wcout.precision() == 7);
  assert(std::wcout.fill() == L'*');
  assert(std::wcin.tie() == &std::wcout);
  assert(std::wcerr.tie() == &std::wcout);
  assert(std::wcerr.flags() & std::ios_base::unitbuf);
  assert(std::wcout.good());
#endif

  return 0;
}
