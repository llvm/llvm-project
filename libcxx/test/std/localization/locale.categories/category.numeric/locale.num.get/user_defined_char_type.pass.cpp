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

struct Char {
  Char() = default;
  Char(char c) : underlying_(c) {}
  Char(unsigned i) : underlying_(i) {}
  explicit Char(std::int32_t i) : underlying_(i) {}
  operator std::int32_t() const { return underlying_; }

  char underlying_;
};

template <>
struct std::char_traits<Char> {
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

// This ctype specialization treats all characters as spaces
template <>
class std::ctype<Char> : public locale::facet, public ctype_base {
public:
  using char_type = Char;
  static locale::id id;
  explicit ctype(std::size_t refs = 0) : locale::facet(refs) {}

  bool is(mask m, char_type) const { return m & ctype_base::space; }
  const char_type* is(const char_type* low, const char_type* high, mask* vec) const {
    for (; low != high; ++low)
      *vec++ = ctype_base::space;
    return high;
  }

  const char_type* scan_is(mask m, const char_type* beg, const char_type* end) const {
    for (; beg != end; ++beg)
      if (this->is(m, *beg))
        return beg;
    return end;
  }

  const char_type* scan_not(mask m, const char_type* beg, const char_type* end) const {
    for (; beg != end; ++beg)
      if (!this->is(m, *beg))
        return beg;
    return end;
  }

  char_type toupper(char_type c) const { return c; }
  const char_type* toupper(char_type*, const char_type* end) const { return end; }

  char_type tolower(char_type c) const { return c; }
  const char_type* tolower(char_type*, const char_type* end) const { return end; }

  char_type widen(char c) const { return char_type(c); }
  const char* widen(const char* beg, const char* end, char_type* dst) const {
    for (; beg != end; ++beg, ++dst)
      *dst = char_type(*beg);
    return end;
  }

  char narrow(char_type c, char /*dflt*/) const { return c.underlying_; }
  const char_type* narrow(const char_type* beg, const char_type* end, char /*dflt*/, char* dst) const {
    for (; beg != end; ++beg, ++dst)
      *dst = beg->underlying_;
    return end;
  }
};

std::locale::id std::ctype<Char>::id;

template <>
class std::numpunct<Char> : public locale::facet {
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

std::locale::id std::numpunct<Char>::id;

int main(int, char**) {
  std::locale l(std::locale::classic(), new std::num_get<Char>);

  return 0;
}
