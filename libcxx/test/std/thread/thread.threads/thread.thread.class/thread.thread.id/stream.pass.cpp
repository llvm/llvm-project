//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: no-localization

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <thread>

// class thread::id

// template<class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& out, thread::id id);

#include <thread>
#include <format>
#include <sstream>
#include <cassert>

#include "make_string.h"
#include "test_macros.h"

template <class CharT>
static void test() {
  std::thread::id id0 = std::this_thread::get_id();
  std::basic_ostringstream<CharT> os;
  os << id0;

#if TEST_STD_VER > 20 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_FORMAT)
  // C++23 added a formatter specialization for thread::id.
  // This changed the requirement of ostream to have a
  // [thread.thread.id]/2
  //   The text representation for the character type charT of an object of
  //   type thread::id is an unspecified sequence of charT ...
  // This definition is used for both streaming and formatting.
  //
  // Test whether the output is identical.
  std::basic_string<CharT> s = std::format(MAKE_STRING_VIEW(CharT, "{}"), id0);
  assert(s == os.str());
#endif
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
