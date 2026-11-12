//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STRING_FIND_OF_HELPERS_H
#define SUPPORT_STRING_FIND_OF_HELPERS_H

// Helpers for testing the find_first_of / find_last_of / find_first_not_of /
// find_last_not_of family of basic_string and basic_string_view against a
// reference implementation, for any character and traits type.

#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "test_macros.h"

// An ASCII case-insensitive traits type. Its eq() is not plain equality, so an
// implementation must not compare the raw characters when this is used.
struct ci_char_traits : std::char_traits<char> {
  static constexpr char to_upper(char c) { return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c; }

  static constexpr bool eq(char a, char b) { return to_upper(a) == to_upper(b); }

  static constexpr bool lt(char a, char b) {
    return static_cast<unsigned char>(to_upper(a)) < static_cast<unsigned char>(to_upper(b));
  }

  static constexpr int compare(const char* a, const char* b, std::size_t n) {
    for (; n != 0; --n, ++a, ++b) {
      if (lt(*a, *b))
        return -1;
      if (lt(*b, *a))
        return 1;
    }
    return 0;
  }

  static constexpr const char* find(const char* s, std::size_t n, char c) {
    for (; n != 0; --n, ++s)
      if (eq(*s, c))
        return s;
    return nullptr;
  }
};

enum class FindOf { first_of, last_of, first_not_of, last_not_of };

template <class Traits, class CharT>
TEST_CONSTEXPR_CXX20 std::size_t
reference_find_of(FindOf op, const CharT* p, std::size_t sz, const CharT* s, std::size_t pos, std::size_t n) {
  auto in_set = [&](CharT c) {
    for (std::size_t j = 0; j != n; ++j)
      if (Traits::eq(s[j], c))
        return true;
    return false;
  };
  bool negate = op == FindOf::first_not_of || op == FindOf::last_not_of;
  if (op == FindOf::first_of || op == FindOf::first_not_of) {
    for (std::size_t i = pos; i < sz; ++i)
      if (in_set(p[i]) != negate)
        return i;
  } else {
    for (std::size_t i = pos < sz ? pos + 1 : sz; i > 0; --i)
      if (in_set(p[i - 1]) != negate)
        return i - 1;
  }
  return static_cast<std::size_t>(-1);
}

template <class S>
TEST_CONSTEXPR_CXX20 typename S::size_type call_find_of(
    FindOf op, const S& s, const typename S::value_type* str, typename S::size_type pos, typename S::size_type n) {
  switch (op) {
  case FindOf::first_of:
    return s.find_first_of(str, pos, n);
  case FindOf::last_of:
    return s.find_last_of(str, pos, n);
  case FindOf::first_not_of:
    return s.find_first_not_of(str, pos, n);
  case FindOf::last_not_of:
    return s.find_last_not_of(str, pos, n);
  }
  assert(false);
  return 0;
}

template <class S>
TEST_CONSTEXPR_CXX20 typename S::size_type
call_find_of(FindOf op, const S& s, const S& str, typename S::size_type pos) {
  switch (op) {
  case FindOf::first_of:
    return s.find_first_of(str, pos);
  case FindOf::last_of:
    return s.find_last_of(str, pos);
  case FindOf::first_not_of:
    return s.find_first_not_of(str, pos);
  case FindOf::last_not_of:
    return s.find_last_not_of(str, pos);
  }
  assert(false);
  return 0;
}

template <class CharT>
TEST_CONSTEXPR_CXX20 bool char_value_fits(unsigned long v) {
  return v <= static_cast<unsigned long>(std::numeric_limits<std::make_unsigned_t<CharT> >::max());
}

// Converts a character value to CharT. Values above the range of a signed
// CharT become negative, e.g. 0xFF becomes '\xff' for a signed char.
template <class CharT>
TEST_CONSTEXPR_CXX20 CharT char_from_value(unsigned long v) {
  return static_cast<CharT>(static_cast<std::make_unsigned_t<CharT> >(v));
}

struct FindOfNeedle {
  unsigned long chars[12];
  std::size_t len;
};

// Checks all four functions against the reference implementation on haystacks
// of various lengths containing a "special" character at various positions,
// with a variety of needles and positions.
//
// The characters include high bytes, embedded '\0', case variants and, for
// wide character types, values of 256 and above (up to the largest value of
// CharT, which is negative for a signed wchar_t). Needles and haystacks mix
// small and large values. Inputs containing values that don't fit in CharT are
// skipped.
//
// The haystack lengths and match positions are chosen around 16, so that
// implementations which check a short prefix before switching to another
// algorithm are tested on both sides of the switch, forwards and backwards.
// The needles include one longer than the rest of most haystacks.
//
// When `full` is false, a reduced set of inputs is used so that the test is
// cheap enough for constant evaluation.
template <class S>
TEST_CONSTEXPR_CXX20 void test_find_of_family(bool full) {
  using CharT  = typename S::value_type;
  using Traits = typename S::traits_type;
  using Buf    = std::basic_string<CharT, Traits>;

  const FindOfNeedle needles[] = {
      {{}, 0},
      {{'Q'}, 1},
      {{'q'}, 1},
      {{0xFF}, 1},
      {{'Q', 'Z'}, 2},
      {{'a', 'Q'}, 2},
      {{'b', 'c'}, 2},
      {{'a', 'b', 'c'}, 3},
      {{'a', 'a', 'a', 'b'}, 4},
      {{0xFF, 0x80}, 2},
      {{0, 'Q'}, 2},
      {{'z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q', 'Q'}, 11},
      {{0x100, 'Q'}, 2},
      {{0xFF, 0x100}, 2},
      {{'a', 0x20AC}, 2},
      {{0x20AC, 0xFF0C}, 2},
      {{0xFFFF, 'b'}, 2},
      {{0x10FFFF, 'b', 'Q'}, 3},
      {{0xFFFFFFFF, 'Q'}, 2},
      {{'a', 0x4E2D}, 2},
      {{0x4E2D, 'b', 'c'}, 3},
  };
  const unsigned long fillers[][3]     = {{'a', 'b', 'c'}, {'a', 0x4E2D, 'c'}};
  const unsigned long specials[]       = {'Q', 'q', 0x80, 0xFF, 0, 0x100, 0x20AC, 0xFF0C, 0xFFFF, 0x10FFFF, 0xFFFFFFFF};
  const unsigned long short_specials[] = {'q', 0xFF0C};
  const std::size_t lengths[]          = {0, 1, 2, 15, 16, 17, 18, 31, 32, 33, 34, 64, 100};
  const std::size_t short_lengths[]    = {0, 33};
  const FindOf ops[]                   = {FindOf::first_of, FindOf::last_of, FindOf::first_not_of, FindOf::last_not_of};

  std::size_t filler_count           = full ? 2 : 1;
  const unsigned long* special_first = full ? specials : short_specials;
  std::size_t special_count =
      full ? sizeof(specials) / sizeof(specials[0]) : sizeof(short_specials) / sizeof(short_specials[0]);
  const std::size_t* length_first = full ? lengths : short_lengths;
  std::size_t length_count =
      full ? sizeof(lengths) / sizeof(lengths[0]) : sizeof(short_lengths) / sizeof(short_lengths[0]);

  for (std::size_t fi = 0; fi != filler_count; ++fi) {
    const unsigned long* filler = fillers[fi];
    if (!char_value_fits<CharT>(filler[1]))
      continue;
    for (std::size_t li = 0; li != length_count; ++li) {
      std::size_t len = length_first[li];

      // Positions near the start and near the end of the haystack, and npos.
      std::size_t positions[48] = {};
      std::size_t pos_count     = 0;
      if (full) {
        for (std::size_t pos = 0; pos <= len + 1; ++pos)
          if (pos <= 20 || pos + 20 >= len)
            positions[pos_count++] = pos;
      } else {
        positions[pos_count++] = 0;
        positions[pos_count++] = 16;
        positions[pos_count++] = len == 0 ? 0 : len - 1;
      }
      positions[pos_count++] = S::npos;

      // `at == len` means the special character is not present.
      const std::size_t ats[] = {0, 15, 16, 17, 18, 32, 33, len - 1, len};
      for (std::size_t si = 0; si != special_count; ++si) {
        if (!char_value_fits<CharT>(special_first[si]))
          continue;
        for (std::size_t at : ats) {
          if (at > len)
            continue;
          Buf hay;
          for (std::size_t i = 0; i != len; ++i)
            hay.push_back(char_from_value<CharT>(filler[i % 3]));
          if (at < len)
            hay[at] = char_from_value<CharT>(special_first[si]);
          S s(hay.data(), hay.size());

          const std::size_t needle_count = sizeof(needles) / sizeof(needles[0]);
          // The last iteration uses a long needle: 40 'z' followed by "Qq".
          for (std::size_t ni = 0; ni <= needle_count; ++ni) {
            Buf nd;
            if (ni == needle_count) {
              if (!full)
                break;
              nd.assign(40, char_from_value<CharT>('z'));
              nd.push_back(char_from_value<CharT>('Q'));
              nd.push_back(char_from_value<CharT>('q'));
            } else {
              const FindOfNeedle& needle = needles[ni];
              bool fits                  = true;
              for (std::size_t j = 0; j != needle.len; ++j)
                fits = fits && char_value_fits<CharT>(needle.chars[j]);
              if (!fits)
                continue;
              for (std::size_t j = 0; j != needle.len; ++j)
                nd.push_back(char_from_value<CharT>(needle.chars[j]));
            }
            S nds(nd.data(), nd.size());

            for (std::size_t pi = 0; pi != pos_count; ++pi) {
              std::size_t pos = positions[pi];
              for (FindOf op : ops) {
                std::size_t expected = reference_find_of<Traits>(op, hay.data(), hay.size(), nd.data(), pos, nd.size());
                assert(call_find_of(op, s, nd.data(), pos, nd.size()) == expected);
                assert(call_find_of(op, s, nds, pos) == expected);
              }
            }
          }
        }
      }
    }
  }
}

// Checks hand-computed results for ci_char_traits. S must be
// basic_string<char, ci_char_traits> or basic_string_view<char, ci_char_traits>.
template <class S>
TEST_CONSTEXPR_CXX20 void test_find_of_case_insensitive() {
  S s("hello WORLD");
  // h0 e1 l2 l3 o4 ' '5 W6 O7 R8 L9 D10
  assert(s.find_first_of("w") == 6);
  assert(s.find_first_of("wd") == 6);
  assert(s.find_first_of("xyzD") == 10);
  assert(s.find_last_of("h") == 0);
  assert(s.find_last_of("lh") == 9);
  assert(s.find_last_of("lh", 8) == 3);
  assert(s.find_first_not_of("H") == 1);
  assert(s.find_first_not_of("HEL") == 4);
  assert(s.find_last_not_of("d") == 9);
  assert(s.find_last_not_of("dlr") == 7);

  // Matches past the first 16 characters.
  S long_s("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxAbxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
  // 'A' is at index 40 and 'b' at index 41; the string has length 82.
  assert(long_s.find_first_of("ab") == 40);
  assert(long_s.find_first_of("B") == 41);
  assert(long_s.find_last_of("ab") == 41);
  assert(long_s.find_last_of("a") == 40);
  assert(long_s.find_first_not_of("X") == 40);
  assert(long_s.find_first_not_of("XA") == 41);
  assert(long_s.find_last_not_of("X") == 41);
  assert(long_s.find_last_not_of("xB") == 40);
}

#endif // SUPPORT_STRING_FIND_OF_HELPERS_H
