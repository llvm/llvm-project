//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that we can explicitly instantiate std::string with a custom
// character type and traits and then use `shrink_to_fit`. In particular, this is
// a regression test for the bug that was reported at https://stackoverflow.com/q/69520633/627587
// and https://seedcentral.apple.com/sm/feedback_collector/radar/85053279.

// RUN: %{cxx} %{flags} %{compile_flags} %s %{link_flags} -DTU1 -c -o %t.tu1.o
// RUN: %{cxx} %{flags} %{compile_flags} %s %{link_flags} -DTU2 -c -o %t.tu2.o
// RUN: %{cxx} %{flags} %t.tu1.o %t.tu2.o %{link_flags} -o %t.exe

// UNSUPPORTED: no-localization

#include <cstdint>
#include <ios>
#include <string>

typedef std::uint16_t char16;

struct string16_char_traits {
  typedef char16 char_type;
  typedef int int_type;

  typedef std::streamoff off_type;
  typedef std::mbstate_t state_type;
  typedef std::fpos<state_type> pos_type;

  static void assign(char_type&, const char_type&) { }
  static bool eq(const char_type&, const char_type&) { return false; }
  static bool lt(const char_type&, const char_type&) { return false; }
  static int compare(const char_type*, const char_type*, std::size_t) { return 0; }
  static std::size_t length(const char_type*) { return 0; }
  static const char_type* find(const char_type*, std::size_t, const char_type&) { return nullptr; }
  static char_type* move(char_type*, const char_type*, std::size_t) { return nullptr; }
  static char_type* copy(char_type*, const char_type*, std::size_t) { return nullptr; }
  static char_type* assign(char_type*, std::size_t, char_type) { return nullptr; }
  static int_type not_eof(const int_type&) { return 0; }
  static char_type to_char_type(const int_type&) { return char_type(); }
  static int_type to_int_type(const char_type&) { return int_type(); }
  static bool eq_int_type(const int_type&, const int_type&) { return false; }
  static int_type eof() { return int_type(); }
};

#if defined(TU1)
template class std::basic_string<char16, string16_char_traits>;
#else
extern template class std::basic_string<char16, string16_char_traits>;

int main(int, char**) {
    std::basic_string<char16, string16_char_traits> s;
    s.shrink_to_fit();
}
#endif
