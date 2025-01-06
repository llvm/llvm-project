//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: libcpp-has-no-unicode

// <format>

// This test the debug string type for the formatter specializations for char
// and string types. This tests ASCII strings, the tests assume every char32_t value is valid ASCII.

#include <cassert>
#include <concepts>
#include <iterator>
#include <list>
#include <vector>

#include "test_macros.h"
#include "make_string.h"
#include "test_format_string.h"
#include "assert_macros.h"
#include "concat_macros.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <iostream>
#endif

#define SV(S) MAKE_STRING_VIEW(CharT, S)

auto test_format = []<class CharT, class... Args>(
                       std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
  {
    std::basic_string<CharT> out = std::format(fmt, std::forward<Args>(args)...);
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  }
#ifndef TEST_HAS_NO_LOCALIZATION
  {
    std::basic_string<CharT> out = std::format(std::locale(), fmt, std::forward<Args>(args)...);
    assert(out == expected);
  }
#endif // TEST_HAS_NO_LOCALIZATION
};

auto test_format_to =
    []<class CharT, class... Args>(
        std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
      {
        std::basic_string<CharT> out(expected.size(), CharT(' '));
        auto it = std::format_to(out.begin(), fmt, std::forward<Args>(args)...);
        assert(it == out.end());
        assert(out == expected);
      }
#ifndef TEST_HAS_NO_LOCALIZATION
      {
        std::basic_string<CharT> out(expected.size(), CharT(' '));
        auto it = std::format_to(out.begin(), std::locale(), fmt, std::forward<Args>(args)...);
        assert(it == out.end());
        assert(out == expected);
      }
#endif // TEST_HAS_NO_LOCALIZATION
      {
        std::list<CharT> out;
        std::format_to(std::back_inserter(out), fmt, std::forward<Args>(args)...);
        assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
      }
      {
        std::vector<CharT> out;
        std::format_to(std::back_inserter(out), fmt, std::forward<Args>(args)...);
        assert(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
      }
      {
        assert(expected.size() < 4096 && "Update the size of the buffer.");
        CharT out[4096];
        CharT* it = std::format_to(out, fmt, std::forward<Args>(args)...);
        assert(std::distance(out, it) == int(expected.size()));
        // Convert to std::string since output contains '\0' for boolean tests.
        assert(std::basic_string<CharT>(out, it) == expected);
      }
    };

auto test_formatted_size =
    []<class CharT, class... Args>(
        std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
      {
        std::size_t size = std::formatted_size(fmt, std::forward<Args>(args)...);
        assert(size == expected.size());
      }
#ifndef TEST_HAS_NO_LOCALIZATION
      {
        std::size_t size = std::formatted_size(std::locale(), fmt, std::forward<Args>(args)...);
        assert(size == expected.size());
      }
#endif // TEST_HAS_NO_LOCALIZATION
    };

auto test_format_to_n =
    []<class CharT, class... Args>(
        std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args) {
      {
        std::size_t n = expected.size();
        std::basic_string<CharT> out(n, CharT(' '));
        std::format_to_n_result result = std::format_to_n(out.begin(), n, fmt, std::forward<Args>(args)...);
        assert(result.size == static_cast<std::ptrdiff_t>(expected.size()));
        assert(result.out == out.end());
        assert(out == expected);
      }
#ifndef TEST_HAS_NO_LOCALIZATION
      {
        std::size_t n = expected.size();
        std::basic_string<CharT> out(n, CharT(' '));
        std::format_to_n_result result =
            std::format_to_n(out.begin(), n, std::locale(), fmt, std::forward<Args>(args)...);
        assert(result.size == static_cast<std::ptrdiff_t>(expected.size()));
        assert(result.out == out.end());
        assert(out == expected);
      }
#endif // TEST_HAS_NO_LOCALIZATION
      {
        std::ptrdiff_t n = 0;
        std::basic_string<CharT> out;
        std::format_to_n_result result = std::format_to_n(out.begin(), n, fmt, std::forward<Args>(args)...);
        assert(result.size == static_cast<std::ptrdiff_t>(expected.size()));
        assert(result.out == out.end());
        assert(out.empty());
      }
      {
        std::ptrdiff_t n = expected.size() / 2;
        std::basic_string<CharT> out(n, CharT(' '));
        std::format_to_n_result result = std::format_to_n(out.begin(), n, fmt, std::forward<Args>(args)...);
        assert(result.size == static_cast<std::ptrdiff_t>(expected.size()));
        assert(result.out == out.end());
        assert(out == expected.substr(0, n));
      }
    };

template <class CharT>
void test_char() {
  // *** P2286 examples ***
  test_format(SV("['\\'', '\"']"), SV("[{:?}, {:?}]"), CharT('\''), CharT('"'));

  // *** Special cases ***
  test_format(SV("'\\t'"), SV("{:?}"), CharT('\t'));
  test_format(SV("'\\n'"), SV("{:?}"), CharT('\n'));
  test_format(SV("'\\r'"), SV("{:?}"), CharT('\r'));
  test_format(SV("'\\\\'"), SV("{:?}"), CharT('\\'));

  test_format(SV("'\\\''"), SV("{:?}"), CharT('\''));
  test_format(SV("'\"'"), SV("{:?}"), CharT('"')); // only special for string

  test_format(SV("' '"), SV("{:?}"), CharT(' '));

  // *** Printable ***
  test_format(SV("'a'"), SV("{:?}"), CharT('a'));
  test_format(SV("'b'"), SV("{:?}"), CharT('b'));
  test_format(SV("'c'"), SV("{:?}"), CharT('c'));

  // *** Non-printable ***

  // Control
  test_format(SV("'\\u{0}'"), SV("{:?}"), CharT('\0'));
  test_format(SV("'\\u{1f}'"), SV("{:?}"), CharT('\x1f'));

  // Ill-formed
  if constexpr (sizeof(CharT) == 1)
    test_format(SV("'\x80'"), SV("{:?}"), CharT('\x80'));

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  if constexpr (sizeof(CharT) > 1) {
    using V = std::basic_string_view<CharT>;

    // Unicode fitting in a 16-bit wchar_t

    // *** Non-printable ***

    // Space_Separator
    test_format(V{L"'\xa0'"}, L"{:?}", L'\xa0');     // NO-BREAK SPACE
    test_format(V{L"'\x3000'"}, L"{:?}", L'\x3000'); // IDEOGRAPHIC SPACE

    // Line_Separator
    test_format(V{L"'\x2028'"}, L"{:?}", L'\x2028'); // LINE SEPARATOR

    // Paragraph_Separator
    test_format(V{L"'\x2029'"}, L"{:?}", L'\x2029'); // PARAGRAPH SEPARATOR

    // Format
    test_format(V{L"'\xad'"}, L"{:?}", L'\xad');     // SOFT HYPHEN
    test_format(V{L"'\x600'"}, L"{:?}", L'\x600');   // ARABIC NUMBER SIGN
    test_format(V{L"'\xfeff'"}, L"{:?}", L'\xfeff'); // ZERO WIDTH NO-BREAK SPACE

    if constexpr (sizeof(CharT) == 2) {
      // Incomplete surrogate pair in UTF-16
      test_format(V{L"'\xd800'"}, L"{:?}", L'\xd800'); // <surrogate-D800>
      test_format(V{L"'\xdfff'"}, L"{:?}", L'\xdfff'); // <surrogate-DFFF>
    } else {
      test_format(V{L"'\xd800'"}, L"{:?}", L'\xd800'); // <surrogate-D800>
      test_format(V{L"'\xdfff'"}, L"{:?}", L'\xdfff'); // <surrogate-DFFF>
    }

    // Private_Use
    test_format(V{L"'\xe000'"}, L"{:?}", L'\xe000'); // <private-use-E000>
    test_format(V{L"'\xf8ff'"}, L"{:?}", L'\xf8ff'); // <private-use-F8FF>

    // Unassigned
    test_format(V{L"'\x378'"}, L"{:?}", L'\x378');   // <reserved-0378>
    test_format(V{L"'\x1774'"}, L"{:?}", L'\x1774'); // <reserved-1774>
    test_format(V{L"'\xffff'"}, L"{:?}", L'\xffff'); // <noncharacter-FFFF>

    // Grapheme Extended
    test_format(V{L"'\x300'"}, L"{:?}", L'\x300');   // COMBINING GRAVE ACCENT
    test_format(V{L"'\xfe20'"}, L"{:?}", L'\xfe20'); // VARIATION SELECTOR-1
  }
#  ifndef TEST_SHORT_WCHAR
  if constexpr (sizeof(CharT) > 2) {
    static_assert(sizeof(CharT) == 4, "add support for unexpected size");
    // Unicode fitting in a 32-bit wchar_t

    constexpr wchar_t x       = 0x1ffff;
    constexpr std::uint32_t y = 0x1ffff;
    static_assert(x == y);

    using V = std::basic_string_view<CharT>;

    // *** Non-printable ***
    // Format
    test_format(V{L"'\x110bd'"}, L"{:?}", L'\x110bd'); // KAITHI NUMBER SIGN
    test_format(V{L"'\xe007f'"}, L"{:?}", L'\xe007f'); // CANCEL TAG

    // Private_Use
    test_format(V{L"'\xf0000'"}, L"{:?}", L'\xf0000'); // <private-use-F0000>
    test_format(V{L"'\xffffd'"}, L"{:?}", L'\xffffd'); // <private-use-FFFFD>

    test_format(V{L"'\x100000'"}, L"{:?}", L'\x100000'); // <private-use-100000>
    test_format(V{L"'\x10fffd'"}, L"{:?}", L'\x10fffd'); // <private-use-10FFFD>

    // Unassigned
    test_format(V{L"'\x1000c'"}, L"{:?}", L'\x1000c');   // <reserved-1000c>
    test_format(V{L"'\xfffff'"}, L"{:?}", L'\xfffff');   // <noncharacter-FFFFF>
    test_format(V{L"'\x10fffe'"}, L"{:?}", L'\x10fffe'); // <noncharacter-10FFFE>

    // Grapheme Extended
    test_format(V{L"'\x101fd'"}, L"{:?}", L'\x101fd'); // COMBINING OLD PERMIC LETTER AN
    test_format(V{L"'\xe0100'"}, L"{:?}", L'\xe0100'); // VARIATION SELECTOR-17

    // Ill-formed
    test_format(V{L"'\x110000'"}, L"{:?}", L'\x110000');
    test_format(V{L"'\xffffffff'"}, L"{:?}", L'\xffffffff');
  }
#  endif // TEST_SHORT_WCHAR
#endif   // TEST_HAS_NO_WIDE_CHARACTERS
}

template <class CharT>
void test_string() {
  // *** P2286 examples ***
  test_format(SV("[h\tllo]"), SV("[{}]"), SV("h\tllo"));
  test_format(SV(R"(["h\tllo"])"), SV("[{:?}]"), SV("h\tllo"));
  test_format(SV(R"(["Спасибо, Виктор ♥!"])"), SV("[{:?}]"), SV("Спасибо, Виктор ♥!"));

  test_format(SV(R"(["\u{0} \n \t \u{2} \u{1b}"])"), SV("[{:?}]"), SV("\0 \n \t \x02 \x1b"));

  if constexpr (sizeof(CharT) == 1) {
    // Ill-formend UTF-8, but valid as ASCII
    test_format(SV("[\"\xc3\"]"), SV("[{:?}]"), SV("\xc3"));
    test_format(SV("[\"\xc3\x28\"]"), SV("[{:?}]"), SV("\xc3\x28"));
  } else {
    // Valid UTF-16 and UTF-32
    test_format(SV("[\"\u00c3\"]"), SV("[{:?}]"), L"\xc3"); // LATIN CAPITAL LETTER A WITH TILDE
    test_format(SV("[\"\u00c3(\"]"), SV("[{:?}]"), L"\xc3\x28");
  }

  test_format(SV("[\"🤷🏻\u200d♂\ufe0f\"]"), SV("[{:?}]"), SV("🤷🏻‍♂️"));

  // *** Special cases ***
  test_format(SV(R"("\t\n\r\\'\" ")"), SV("{:?}"), SV("\t\n\r\\'\" "));

  // *** Printable ***
  test_format(SV(R"("abcdefg")"), SV("{:?}"), SV("abcdefg"));

  // *** Non-printable ***

  // Control
  test_format(SV(R"("\u{0}\u{1f}")"), SV("{:?}"), SV("\0\x1f"));

  // Ill-formed UTF-8, valid ASCII
  test_format(SV("\"\x80\""), SV("{:?}"), SV("\x80"));
}

template <class CharT, class TestFunction>
void test_format_functions(TestFunction check) {
  // LATIN SMALL LETTER O WITH DIAERESIS is encoded in two chars or 1 wchar_t
  // due to the range of the value.
  // 8 + sizeof(CharT) == 1 is not considered an constant expression

  // *** align-fill & width ***
  check(SV(R"(***"hellö")"),
        SV("{:*>{}?}"),
        SV("hellö"),
        sizeof(CharT) == 1 ? 11 : 10); // ö is LATIN SMALL LETTER O WITH DIAERESIS
  check(SV(R"(*"hellö"**)"), SV("{:*^{}?}"), SV("hellö"), sizeof(CharT) == 1 ? 11 : 10);
  check(SV(R"("hellö"***)"), SV("{:*<{}?}"), SV("hellö"), sizeof(CharT) == 1 ? 11 : 10);

  check(SV("\"hello\u0308\""), SV("{:*>{}?}"), SV("hello\u0308"), sizeof(CharT) == 1 ? 9 : 8);
  check(SV("***\"hello\u0308\""), SV("{:*>{}?}"), SV("hello\u0308"), sizeof(CharT) == 1 ? 12 : 11);
  check(SV("*\"hello\u0308\"**"), SV("{:*^{}?}"), SV("hello\u0308"), sizeof(CharT) == 1 ? 12 : 11);
  check(SV("\"hello\u0308\"***"), SV("{:*<{}?}"), SV("hello\u0308"), sizeof(CharT) == 1 ? 12 : 11);

  // *** width ***
  check(SV(R"("hello"   )"), SV("{:10?}"), SV("hello"));

  // *** precision ***
  check(SV(R"("hell)"), SV("{:.5?}"), SV("hello"));
  check(SV(R"("hello)"), SV("{:.6?}"), SV("hello"));
  check(SV(R"("hello")"), SV("{:.7?}"), SV("hello"));

  // *** width & precision ***
  check(SV(R"("hell#########################)"), SV("{:#<30.5?}"), SV("hello"));
  check(SV(R"("hello########################)"), SV("{:#<30.6?}"), SV("hello"));
  check(SV(R"("hello"#######################)"), SV("{:#<30.7?}"), SV("hello"));
}

template <class CharT>
void test() {
  test_char<CharT>();

  test_string<CharT>();

  test_format_functions<CharT>(test_format);
  test_format_functions<CharT>(test_format_to);
  test_format_functions<CharT>(test_formatted_size);
  test_format_functions<CharT>(test_format_to_n);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
