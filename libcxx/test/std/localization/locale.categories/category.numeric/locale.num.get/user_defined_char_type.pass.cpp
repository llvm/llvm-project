//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>
//
// Make sure that num_get works with a user-defined char_type that has a
// constructor making initialization from bare `int` invalid.

#include <cstddef>
#include <cstdint>
#include <locale>
#include <string>

#include "test_macros.h"

struct Char {
  Char() = default;
  Char(char c) : underlying_(c) {}
  Char(unsigned i) : underlying_(i) {}
  explicit Char(std::int32_t i) : underlying_(i) {}
  operator std::int32_t() const { return underlying_; }

  char underlying_;
};

namespace std {
template <>
struct char_traits<Char> {
  using char_type  = Char;
  using int_type   = int;
  using off_type   = streamoff;
  using pos_type   = streampos;
  using state_type = mbstate_t;

  static void assign(char_type& a, const char_type& b) { a = b; }
  static bool eq(char_type a, char_type b) { return a.underlying_ == b.underlying_; }
  static bool lt(char_type a, char_type b) { return a.underlying_ < b.underlying_; }

  static int compare(const char_type* s1, const char_type* s2, std::size_t n) {
    return char_traits<char>::compare(reinterpret_cast<const char*>(s1), reinterpret_cast<const char*>(s2), n);
  }
  static std::size_t length(const char_type* s) { return char_traits<char>::length(reinterpret_cast<const char*>(s)); }
  static const char_type* find(const char_type* p, std::size_t n, const char_type& c) {
    for (size_t i = 0; i != n; ++i) {
      if (static_cast<int32_t>(p[i]) == static_cast<int32_t>(c)) {
        return p + n;
      }
    }
    return nullptr;
  }
  static char_type* move(char_type* dest, const char_type* source, std::size_t count) {
    char_traits<char>::move(reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(source), count);
    return dest;
  }
  static char_type* copy(char_type* dest, const char_type* source, std::size_t count) {
    char_traits<char>::copy(reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(source), count);
    return dest;
  }
  static char_type* assign(char_type* dest, std::size_t n, char_type c) {
    char_traits<char>::assign(reinterpret_cast<char*>(dest), n, c.underlying_);
    return dest;
  }

  static int_type not_eof(int_type i) { return char_traits<char>::not_eof(i); }
  static char_type to_char_type(int_type i) { return Char(char_traits<char>::to_char_type(i)); }
  static int_type to_int_type(char_type c) { return char_traits<char>::to_int_type(c.underlying_); }
  static bool eq_int_type(int_type i, int_type j) { return i == j; }
  static int_type eof() { return char_traits<char>::eof(); }
};

template <>
class ctype<Char> : public locale::facet {
public:
  static locale::id id;
  Char toupper(Char c) const { return Char(std::toupper(c.underlying_)); }
  const char* widen(const char* first, const char* last, Char* dst) const {
    for (; first != last;)
      *dst++ = Char(*first++);
    return last;
  }
};

locale::id ctype<Char>::id;

template <>
class numpunct<Char> : public locale::facet {
public:
  typedef basic_string<Char> string_type;
  static locale::id id;
  Char decimal_point() const { return Char('.'); }
  Char thousands_sep() const { return Char(','); }
  string grouping() const { return ""; }
  string_type truename() const {
    static Char yes[] = {Char('t')};
    return string_type(yes, 1);
  }
  string_type falsename() const {
    static Char no[] = {Char('f')};
    return string_type(no, 1);
  }
};

locale::id numpunct<Char>::id;

} // namespace std

int main(int, char**) {
  std::locale l(std::locale::classic(), new std::num_get<Char>);

  return 0;
}
