//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_SUPPORT_LOCALE_FACETS_H
#define LIBCXX_TEST_SUPPORT_LOCALE_FACETS_H

#include <algorithm>
#include <cstring>
#include <locale>

namespace facet {
struct state_t {
  char next_char;
  bool has_next_char;
  bool expect_at_least_5;
};

struct char_traits {
  using base = std::char_traits<char>;

  using char_type  = base::char_type;
  using int_type   = base::int_type;
  using off_type   = base::off_type;
  using pos_type   = std::fpos<state_t>;
  using state_type = state_t;

  static void assign(char_type& lhs, char_type& rhs) { lhs = rhs; }
  static bool eq(char_type lhs, char_type rhs) { return lhs == rhs; }
  static bool lt(char_type lhs, char_type rhs) {
    return static_cast<unsigned char>(lhs) < static_cast<unsigned char>(rhs);
  }

  static int compare(const char_type* lhs, const char_type* rhs, size_t n) { return std::memcmp(lhs, rhs, n); }

  static size_t length(const char_type* str) { return std::strlen(str); }

  static const char_type* find(const char_type* str, size_t count, const char_type& c) {
    return static_cast<const char_type*>(std::memchr(str, c, count));
  }

  static char_type* move(char_type* dst, const char_type* src, size_t count) {
    return static_cast<char_type*>(std::memmove(dst, src, count));
  }

  static char_type* copy(char_type* dst, const char_type* src, size_t count) {
    return static_cast<char_type*>(std::memcpy(dst, src, count));
  }

  static char_type* assign(char_type* dst, size_t count, char_type c) {
    return static_cast<char_type*>(std::memset(dst, c, count));
  }

  static int_type not_eof(int_type c) { return eq_int_type(c, eof()) ? ~eof() : c; }
  static char_type to_char_type(int_type c) { return c; }
  static int_type to_int_type(char_type c) { return c; }
  static bool eq_int_type(int_type lhs, int_type rhs) { return lhs == rhs; }
  static int_type eof() { return EOF; }
};
} // namespace facet

template <>
class std::codecvt<char, char, facet::state_t> : public std::locale::facet, public std::codecvt_base {
public:
  using intern_type = char;
  using extern_type = char;
  using state_type  = ::facet::state_t;

  static inline locale::id id;

  explicit codecvt(size_t refs = 0) : locale::facet(refs) {}

  result out(state_type& state,
             const intern_type* from_first,
             const intern_type* from_last,
             const intern_type*& from_next,
             extern_type* to_first,
             extern_type* to_last,
             extern_type*& to_next) const {
    return do_out(state, from_first, from_last, from_next, to_first, to_last, to_next);
  }

  result unshift(state_type& state, extern_type* to_first, extern_type* to_last, extern_type*& to_next) const {
    return do_unshift(state, to_first, to_last, to_next);
  }

  result in(state_type& state,
            const extern_type* from_first,
            const extern_type* from_last,
            const extern_type*& from_next,
            intern_type* to_first,
            intern_type* to_last,
            intern_type*& to_next) const {
    return do_in(state, from_first, from_last, from_next, to_first, to_last, to_next);
  }

  int encoding() const { return do_encoding(); }
  bool always_noconv() const { return do_always_noconv(); }
  int length(state_type& state, const extern_type* from_first, const extern_type* from_last, size_t max) const {
    return do_length(state, from_first, from_last, max);
  }

  int max_length() const { return do_max_length(); }

protected:
  virtual result
  do_out(state_type& state,
         const intern_type* from_first,
         const intern_type* from_last,
         const intern_type*& from_next,
         extern_type* to_first,
         extern_type* to_last,
         extern_type*& to_next) const = 0;

  virtual result
  do_unshift(state_type& state, extern_type* to_first, extern_type* to_last, extern_type*& to_next) const = 0;

  virtual result
  do_in(state_type& state,
        const extern_type* from_first,
        const extern_type* from_last,
        const extern_type*& from_next,
        intern_type* to_first,
        intern_type* to_last,
        intern_type*& to_next) const = 0;

  virtual int do_encoding() const       = 0;
  virtual bool do_always_noconv() const = 0;
  virtual int
  do_length(state_type& state, const extern_type* from_first, const extern_type* from_last, size_t max) const = 0;
  virtual int do_max_length() const                                                                           = 0;
};

namespace facet {

using codecvt_base_t = std::codecvt<char, char, facet::state_t>;

class codecvt_constant_converting final : public codecvt_base_t {
public:
  template <bool ToExtern>
  static result do_conversion(
      const char* from_first,
      const char* from_last,
      const char*& from_next,
      char* to_first,
      char* to_last,
      char*& to_next) {
    auto size = std::min(from_last - from_first, to_last - to_first);
    std::transform(from_first, from_first + size, to_first, [](unsigned char c) -> char {
      return ToExtern ? c + 30 : c - 30;
    });
    from_next = from_first + size;
    to_next   = to_first + size;
    return size == from_last - from_first ? ok : partial;
  }

  result do_out(state_type&,
                const intern_type* from_first,
                const intern_type* from_last,
                const intern_type*& from_next,
                extern_type* to_first,
                extern_type* to_last,
                extern_type*& to_next) const override {
    return do_conversion<true>(from_first, from_last, from_next, to_first, to_last, to_next);
  }

  result
  do_unshift(state_type&, extern_type* to_first, extern_type* /*to_last*/, extern_type*& to_next) const override {
    to_next = to_first;
    return noconv;
  }

  result do_in(state_type&,
               const extern_type* from_first,
               const extern_type* from_last,
               const extern_type*& from_next,
               intern_type* to_first,
               intern_type* to_last,
               intern_type*& to_next) const override {
    return do_conversion<false>(from_first, from_last, from_next, to_first, to_last, to_next);
  }

  int do_encoding() const override { return 1; }
  bool do_always_noconv() const override { return false; }
  int do_length(state_type&, const extern_type* from_first, const extern_type* from_last, size_t max) const override {
    return static_cast<int>(std::min(max, static_cast<size_t>(from_last - from_first)));
  }
  int do_max_length() const override { return 1; }
};

class codecvt_variable_converting final : public codecvt_base_t {
  result do_out(state_type& state,
                const intern_type* from_first,
                const intern_type* from_last,
                const intern_type*& from_next,
                extern_type* to_first,
                extern_type* to_last,
                extern_type*& to_next) const override {
    for (; from_first != from_last && to_first != to_last; ++from_first, ++to_first) {
      if (*from_first == -1) {
        state.expect_at_least_5 = true;
        continue;
      }
      char c = *from_first;
      if (state.expect_at_least_5 ? c < '5' : c >= '5') {
        from_next = from_first;
        to_next = to_first;
        return error;
      }
      *to_first = c;
      state.expect_at_least_5 = false;
    }
    return from_first == from_last ? ok : partial;
  }

  result
  do_unshift(state_type& state, extern_type* to_first, extern_type* /*to_last*/, extern_type*& to_next) const override {
    if (state.expect_at_least_5)
      return error;
    to_next = to_first;
    return noconv;
  }

  result do_in(state_type& state,
               const extern_type* from_first,
               const extern_type* from_last,
               const extern_type*& from_next,
               intern_type* to_first,
               intern_type* to_last,
               intern_type*& to_next) const override {
    if (state.has_next_char) {
      if (to_first != to_last) {
        *to_first++ = state.next_char;
        state.has_next_char = false;
      }
    }
    for (; from_first != from_last && to_first != to_last; ++from_first, ++to_first) {
      if (*from_first >= '5') {
        if (to_first + 1 == to_last) {
          state.has_next_char = true;
          state.next_char = *from_first;
          *to_first = -1;
          break;
        } else {
          *to_first++ = -1;
        }
      }
      *to_first = *from_first;
    }
    from_next = from_first;
    to_next = to_first;
    return from_first == from_last && !state.has_next_char ? ok : partial;
  }

  int do_encoding() const override { return 0; }
  bool do_always_noconv() const override { return false; }
  int do_length(state_type&, const extern_type* from_first, const extern_type* from_last, size_t max) const override {
    return static_cast<int>(std::min(max, static_cast<size_t>(from_last - from_first)));
  }
  int do_max_length() const override { return 2; }
};

template <class CodeCVT>
inline std::locale get_codecvt_locale() {
  return std::locale(std::locale::classic(), new CodeCVT);
}
} // namespace facet

#endif // LIBCXX_TEST_SUPPORT_LOCALE_FACETS_H
