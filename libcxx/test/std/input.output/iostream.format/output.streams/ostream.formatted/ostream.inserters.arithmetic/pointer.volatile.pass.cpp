//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ostream>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// This test fails in Windows DLL configurations, due to
// __exclude_from_explicit_instantiation__ not behaving as it should in
// combination with dllimport (https://llvm.org/PR41018), in combination
// with running tests in c++23 mode while building the library in c++20 mode.
//
// If the library was built in c++23 mode, this test would succeed.
//
// Older CMake passed -std:c++latest to set C++ 20 mode on clang-cl, which
// hid this issue. With newer CMake versions, it passes -std:c++20 which
// makes this fail.
//
// Marking as UNSUPPORTED instead of XFAIL to avoid spurious failures/successes
// depending on the version of CMake used.
// TODO: Remove this when the library is built in c++23 mode.
//
// UNSUPPORTED: windows-dll

// template <class charT, class traits = char_traits<charT> >
//   class basic_ostream;

// basic_ostream& operator<<(const volatile void* val);

#include <ostream>
#include <cassert>

template <class CharT>
class testbuf : public std::basic_streambuf<CharT> {
  typedef std::basic_streambuf<CharT> base;
  std::basic_string<CharT> str_;

public:
  testbuf() {}

  std::basic_string<CharT> str() const { return std::basic_string<CharT>(base::pbase(), base::pptr()); }

protected:
  virtual typename base::int_type overflow(typename base::int_type ch = base::traits_type::eof()) {
    if (ch != base::traits_type::eof()) {
      int n = static_cast<int>(str_.size());
      str_.push_back(static_cast<CharT>(ch));
      str_.resize(str_.capacity());
      base::setp(const_cast<CharT*>(str_.data()), const_cast<CharT*>(str_.data() + str_.size()));
      base::pbump(n + 1);
    }
    return ch;
  }
};

int main(int, char**) {
  testbuf<char> sb1;
  std::ostream os1(&sb1);
  int n1;
  os1 << &n1;
  assert(os1.good());
  std::string s1 = sb1.str();

  testbuf<char> sb2;
  std::ostream os2(&sb2);
  os2 << static_cast<volatile void*>(&n1);
  assert(os2.good());
  std::string s2 = sb2.str();

  testbuf<char> sb3;
  std::ostream os3(&sb3);
  volatile int n3;
  os3 << &n3;
  assert(os3.good());
  std::string s3 = sb3.str();

  // %p is implementation defined. Instead of validating the
  // output, at least ensure that it does not generate an empty
  // string. Also make sure that given two distinct addresses, the
  // output of %p is different.
  assert(!s1.empty());
  assert(!s2.empty());
  assert(s1 == s2);

  assert(!s3.empty());
  assert(s2 != s3);

  return 0;
}
