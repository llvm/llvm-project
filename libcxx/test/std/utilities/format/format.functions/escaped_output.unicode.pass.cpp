//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// This version runs the test when the platform has Unicode support.
// UNSUPPORTED: libcpp-has-no-unicode

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <format>

// This test the debug string type for the formatter specializations for char
// and string types. This tests Unicode strings.

#include <format>

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

  // *** Specical cases ***
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
    test_format(SV("'\\x{80}'"), SV("{:?}"), CharT('\x80'));

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  if constexpr (sizeof(CharT) > 1) {
    using V = std::basic_string_view<CharT>;

    // Unicode fitting in a 16-bit wchar_t

    // *** Non-printable ***

    // Space_Separator
    test_format(V{L"'\\u{a0}'"}, L"{:?}", L'\xa0');     // NO-BREAK SPACE
    test_format(V{L"'\\u{3000}'"}, L"{:?}", L'\x3000'); // IDEOGRAPHIC SPACE

    // Line_Separator
    test_format(V{L"'\\u{2028}'"}, L"{:?}", L'\x2028'); // LINE SEPARATOR

    // Paragraph_Separator
    test_format(V{L"'\\u{2029}'"}, L"{:?}", L'\x2029'); // PARAGRAPH SEPARATOR

    // Format
    test_format(V{L"'\\u{ad}'"}, L"{:?}", L'\xad');     // SOFT HYPHEN
    test_format(V{L"'\\u{600}'"}, L"{:?}", L'\x600');   // ARABIC NUMBER SIGN
    test_format(V{L"'\\u{feff}'"}, L"{:?}", L'\xfeff'); // ZERO WIDTH NO-BREAK SPACE

    // Incomplete surrogate pair in UTF-16
    test_format(V{L"'\\x{d800}'"}, L"{:?}", L'\xd800'); // <surrogate-D800>
    test_format(V{L"'\\x{dfff}'"}, L"{:?}", L'\xdfff'); // <surrogate-DFFF>

    // Private_Use
    test_format(V{L"'\\u{e000}'"}, L"{:?}", L'\xe000'); // <private-use-E000>
    test_format(V{L"'\\u{f8ff}'"}, L"{:?}", L'\xf8ff'); // <private-use-F8FF>

    // Unassigned
    test_format(V{L"'\\u{378}'"}, L"{:?}", L'\x378');   // <reserved-0378>
    test_format(V{L"'\\u{1774}'"}, L"{:?}", L'\x1774'); // <reserved-1774>
    test_format(V{L"'\\u{ffff}'"}, L"{:?}", L'\xffff'); // <noncharacter-FFFF>

    // Grapheme Extended
    test_format(V{L"'\\u{300}'"}, L"{:?}", L'\x300');   // COMBINING GRAVE ACCENT
    test_format(V{L"'\\u{fe20}'"}, L"{:?}", L'\xfe20'); // VARIATION SELECTOR-1
  }
#  ifndef TEST_SHORT_WCHAR
  if constexpr (sizeof(CharT) > 2) {
    static_assert(sizeof(CharT) == 4, "add support for unexpected size");
    // Unicode fitting in a 32-bit wchar_t

    constexpr wchar_t x  = 0x1ffff;
    constexpr std::uint32_t y = 0x1ffff;
    static_assert(x == y);

    using V = std::basic_string_view<CharT>;

    // *** Non-printable ***
    // Format
    test_format(V{L"'\\u{110bd}'"}, L"{:?}", L'\x110bd'); // KAITHI NUMBER SIGN
    test_format(V{L"'\\u{e007f}'"}, L"{:?}", L'\xe007f'); // CANCEL TAG

    // Private_Use
    test_format(V{L"'\\u{f0000}'"}, L"{:?}", L'\xf0000'); // <private-use-F0000>
    test_format(V{L"'\\u{ffffd}'"}, L"{:?}", L'\xffffd'); // <private-use-FFFFD>

    test_format(V{L"'\\u{100000}'"}, L"{:?}", L'\x100000'); // <private-use-100000>
    test_format(V{L"'\\u{10fffd}'"}, L"{:?}", L'\x10fffd'); // <private-use-10FFFD>

    // Unassigned
    test_format(V{L"'\\u{1000c}'"}, L"{:?}", L'\x1000c');   // <reserved-1000c>
    test_format(V{L"'\\u{fffff}'"}, L"{:?}", L'\xfffff');   // <noncharacter-FFFFF>
    test_format(V{L"'\\u{10fffe}'"}, L"{:?}", L'\x10fffe'); // <noncharacter-10FFFE>

    // Grapheme Extended
    test_format(V{L"'\\u{101fd}'"}, L"{:?}", L'\x101fd'); // COMBINING OLD PERMIC LETTER AN
    test_format(V{L"'\\u{e0100}'"}, L"{:?}", L'\xe0100'); // VARIATION SELECTOR-17

    // Ill-formed
    test_format(V{L"'\\x{110000}'"}, L"{:?}", L'\x110000');
    test_format(V{L"'\\x{ffffffff}'"}, L"{:?}", L'\xffffffff');
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
    // Ill-formend UTF-8
    test_format(SV(R"(["\x{c3}"])"), SV("[{:?}]"), "\xc3");
    test_format(SV(R"(["\x{c3}("])"), SV("[{:?}]"), "\xc3\x28");

    /* U+0000..U+0007F 1 code unit range, encoded in 2 code units. */
    test_format(SV(R"(["\x{c0}\x{80}"])"), SV("[{:?}]"), "\xc0\x80"); // U+0000
    test_format(SV(R"(["\x{c1}\x{bf}"])"), SV("[{:?}]"), "\xc1\xbf"); // U+007F
    test_format(SV(R"(["\u{80}"])"), SV("[{:?}]"), "\xc2\x80");       // U+0080 first valid (General_Category=Control)

    /* U+0000..U+07FFF 1 and 2 code unit range, encoded in 3 code units. */
    test_format(SV(R"(["\x{e0}\x{80}\x{80}"])"), SV("[{:?}]"), "\xe0\x80\x80"); // U+0000
    test_format(SV(R"(["\x{e0}\x{81}\x{bf}"])"), SV("[{:?}]"), "\xe0\x81\xbf"); // U+007F
    test_format(SV(R"(["\x{e0}\x{82}\x{80}"])"), SV("[{:?}]"), "\xe0\x82\x80"); // U+0080
    test_format(SV(R"(["\x{e0}\x{9f}\x{bf}"])"), SV("[{:?}]"), "\xe0\x9f\xbf"); // U+07FF
    test_format(SV("[\"\u0800\"]"), SV("[{:?}]"), "\xe0\xa0\x80");              // U+0800 first valid

#if 0
	// This code point is in the Hangul Jamo Extended-B block and at the time of writing
	// it's unassigned. When it comes defined, this branch might become true.
    test_format(SV("[\"\ud7ff\"]"), SV("[{:?}]"), "\xed\x9f\xbf");              // U+D7FF last valid
#else
    /* U+D800..D+DFFFF surrogate range */
    test_format(SV(R"(["\u{d7ff}"])"), SV("[{:?}]"), "\xed\x9f\xbf");           // U+D7FF last valid
#endif
    test_format(SV(R"(["\x{ed}\x{a0}\x{80}"])"), SV("[{:?}]"), "\xed\xa0\x80"); // U+D800
    test_format(SV(R"(["\x{ed}\x{af}\x{bf}"])"), SV("[{:?}]"), "\xed\xaf\xbf"); // U+DBFF
    test_format(SV(R"(["\x{ed}\x{bf}\x{80}"])"), SV("[{:?}]"), "\xed\xbf\x80"); // U+DC00
    test_format(SV(R"(["\x{ed}\x{bf}\x{bf}"])"), SV("[{:?}]"), "\xed\xbf\xbf"); // U+DFFF
    test_format(SV(R"(["\u{e000}"])"), SV("[{:?}]"), "\xee\x80\x80");           // U+E000 first valid
                                                                                // (in the Private Use Area block)

    /* U+0000..U+FFFF 1, 2, and 3 code unit range */
    test_format(SV(R"(["\x{f0}\x{80}\x{80}\x{80}"])"), SV("[{:?}]"), "\xf0\x80\x80\x80"); // U+0000
    test_format(SV(R"(["\x{f0}\x{80}\x{81}\x{bf}"])"), SV("[{:?}]"), "\xf0\x80\x81\xbf"); // U+007F
    test_format(SV(R"(["\x{f0}\x{80}\x{82}\x{80}"])"), SV("[{:?}]"), "\xf0\x80\x82\x80"); // U+0080
    test_format(SV(R"(["\x{f0}\x{80}\x{9f}\x{bf}"])"), SV("[{:?}]"), "\xf0\x80\x9f\xbf"); // U+07FF
    test_format(SV(R"(["\x{f0}\x{80}\x{a0}\x{80}"])"), SV("[{:?}]"), "\xf0\x80\xa0\x80"); // U+0800
    test_format(SV(R"(["\x{f0}\x{8f}\x{bf}\x{bf}"])"), SV("[{:?}]"), "\xf0\x8f\xbf\xbf"); // U+FFFF
    test_format(SV("[\"\U00010000\"]"), SV("[{:?}]"), "\xf0\x90\x80\x80");                // U+10000 first valid

    /* U+10FFFF..U+1FFFFF invalid range */
    test_format(SV(R"(["\u{10ffff}"])"), SV("[{:?}]"), "\xf4\x8f\xbf\xbf"); // U+10FFFF last valid
                                                                            // (in Supplementary Private Use Area-B)
    test_format(SV(R"(["\x{f4}\x{90}\x{80}\x{80}"])"), SV("[{:?}]"), "\xf4\x90\x80\x80"); // U+110000
    test_format(SV(R"(["\x{f4}\x{bf}\x{bf}\x{bf}"])"), SV("[{:?}]"), "\xf4\xbf\xbf\xbf"); // U+11FFFF
  } else {
    // Valid UTF-16 and UTF-32
    test_format(SV("[\"\u00c3\"]"), SV("[{:?}]"), L"\xc3"); // LATIN CAPITAL LETTER A WITH TILDE
    test_format(SV("[\"\u00c3(\"]"), SV("[{:?}]"), L"\xc3\x28");
  }

  test_format(SV(R"(["🤷🏻\u{200d}♂\u{fe0f}"])"), SV("[{:?}]"), SV("🤷🏻‍♂️"));

  // *** Specical cases ***
  test_format(SV(R"("\t\n\r\\'\" ")"), SV("{:?}"), SV("\t\n\r\\'\" "));

  // *** Printable ***
  test_format(SV(R"("abcdefg")"), SV("{:?}"), SV("abcdefg"));

  // *** Non-printable ***

  // Control
  test_format(SV(R"("\u{0}\u{1f}")"), SV("{:?}"), SV("\0\x1f"));

  // Ill-formed
  if constexpr (sizeof(CharT) == 1)
    test_format(SV(R"("\x{80}")"), SV("{:?}"), SV("\x80"));

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  if constexpr (sizeof(CharT) > 1) {
    using V = std::basic_string_view<CharT>;

    // Unicode fitting in a 16-bit wchar_t

    // *** Non-printable ***

    // Space_Separator
    test_format(V{LR"("\u{a0}\u{3000}")"}, L"{:?}", L"\xa0\x3000");

    // Line_Separator
    test_format(V{LR"("\u{2028}")"}, L"{:?}", L"\x2028"); // LINE SEPARATOR

    // Paragraph_Separator
    test_format(V{LR"("\u{2029}")"}, L"{:?}", L"\x2029"); // PARAGRAPH SEPARATOR

    // Format
    test_format(V{LR"("\u{ad}\u{600}\u{feff}")"}, L"{:?}", L"\xad\x600\xfeff");

    // Incomplete surrogate pair in UTF-16
    test_format(V{LR"("\x{d800}")"}, L"{:?}", L"\xd800");

    // Private_Use
    test_format(V{LR"("\u{e000}\u{f8ff}")"}, L"{:?}", L"\xe000\xf8ff");

    // Unassigned
    test_format(V{LR"("\u{378}\u{1774}\u{ffff}")"}, L"{:?}", L"\x378\x1774\xffff");

    // Grapheme Extended
    test_format(V{LR"("\u{300}\u{fe20}")"}, L"{:?}", L"\x300\xfe20");
  }
#  ifndef TEST_SHORT_WCHAR
  if constexpr (sizeof(CharT) > 2) {
    static_assert(sizeof(CharT) == 4, "add support for unexpected size");
    // Unicode fitting in a 32-bit wchar_t

    constexpr wchar_t x  = 0x1ffff;
    constexpr std::uint32_t y = 0x1ffff;
    static_assert(x == y);

    using V = std::basic_string_view<CharT>;

    // *** Non-printable ***
    // Format
    test_format(V{LR"("\u{110bd}\u{e007f}")"}, L"{:?}", L"\x110bd\xe007f");

    // Private_Use
    test_format(V{LR"("\u{f0000}\u{ffffd}\u{100000}\u{10fffd}")"}, L"{:?}", L"\xf0000\xffffd\x100000\x10fffd");

    // Unassigned
    test_format(V{LR"("\u{1000c}\u{fffff}\u{10fffe}")"}, L"{:?}", L"\x1000c\xfffff\x10fffe");

    // Grapheme Extended
    test_format(V{LR"("\u{101fd}\u{e0100}")"}, L"{:?}", L"\x101fd\xe0100");

    // Ill-formed
    test_format(V{LR"("\x{110000}\x{ffffffff}")"}, L"{:?}", L"\x110000\xffffffff");
  }
#  endif // TEST_SHORT_WCHAR
#endif   // TEST_HAS_NO_WIDE_CHARACTERS
}

template <class CharT, class TestFunction>
void test_format_functions(TestFunction check) {
  // *** align-fill & width ***
  check(SV(R"(***"hellö")"), SV("{:*>10?}"), SV("hellö")); // ö is LATIN SMALL LETTER O WITH DIAERESIS
  check(SV(R"(*"hellö"**)"), SV("{:*^10?}"), SV("hellö"));
  check(SV(R"("hellö"***)"), SV("{:*<10?}"), SV("hellö"));

  check(SV(R"("hello\u{308}")"), SV("{:*>10?}"), SV("hello\u0308"));
  check(SV(R"(***"hello\u{308}")"), SV("{:*>17?}"), SV("hello\u0308"));
  check(SV(R"(*"hello\u{308}"**)"), SV("{:*^17?}"), SV("hello\u0308"));
  check(SV(R"("hello\u{308}"***)"), SV("{:*<17?}"), SV("hello\u0308"));

  check(SV(R"("hello 🤷🏻\u{200d}♂\u{fe0f}")"), SV("{:*>10?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"(***"hello 🤷🏻\u{200d}♂\u{fe0f}")"), SV("{:*>30?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"(*"hello 🤷🏻\u{200d}♂\u{fe0f}"**)"), SV("{:*^30?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂\u{fe0f}"***)"), SV("{:*<30?}"), SV("hello 🤷🏻‍♂️"));

  // *** width ***
  check(SV(R"("hellö"   )"), SV("{:10?}"), SV("hellö"));
  check(SV(R"("hello\u{308}"   )"), SV("{:17?}"), SV("hello\u0308"));
  check(SV(R"("hello 🤷🏻\u{200d}♂\u{fe0f}"   )"), SV("{:30?}"), SV("hello 🤷🏻‍♂️"));

  // *** precision ***
  check(SV(R"("hell)"), SV("{:.5?}"), SV("hellö"));
  check(SV(R"("hellö)"), SV("{:.6?}"), SV("hellö"));
  check(SV(R"("hellö")"), SV("{:.7?}"), SV("hellö"));

  check(SV(R"("hello )"), SV("{:.7?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello )"), SV("{:.8?}"), SV("hello 🤷🏻‍♂️")); // shrug is two columns
  check(SV(R"("hello 🤷🏻)"), SV("{:.9?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\)"), SV("{:.10?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d})"), SV("{:.17?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂)"), SV("{:.18?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂\)"), SV("{:.19?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂\u{fe0f}")"), SV("{:.28?}"), SV("hello 🤷🏻‍♂️"));

  // *** width & precision ***
  check(SV(R"("hell#########################)"), SV("{:#<30.5?}"), SV("hellö"));
  check(SV(R"("hellö########################)"), SV("{:#<30.6?}"), SV("hellö"));
  check(SV(R"("hellö"#######################)"), SV("{:#<30.7?}"), SV("hellö"));

  check(SV(R"("hello #######################)"), SV("{:#<30.7?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello #######################)"), SV("{:#<30.8?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻#####################)"), SV("{:#<30.9?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\####################)"), SV("{:#<30.10?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}#############)"), SV("{:#<30.17?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂############)"), SV("{:#<30.18?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂\###########)"), SV("{:#<30.19?}"), SV("hello 🤷🏻‍♂️"));
  check(SV(R"("hello 🤷🏻\u{200d}♂\u{fe0f}"###)"), SV("{:#<30.28?}"), SV("hello 🤷🏻‍♂️"));
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

static void test_ill_formed_utf8() {
  using namespace std::literals;

  // Too few code units
  test_format(R"("\x{df}")"sv, "{:?}", "\xdf");
  test_format(R"("\x{ef}")"sv, "{:?}", "\xef");
  test_format(R"("\x{ef}\x{bf}")"sv, "{:?}", "\xef\xbf");
  test_format(R"("\x{f7}")"sv, "{:?}", "\xf7");
  test_format(R"("\x{f7}\x{bf}")"sv, "{:?}", "\xf7\xbf");
  test_format(R"("\x{f7}\x{bf}\x{bf}")"sv, "{:?}", "\xf7\xbf\xbf");

  // Invalid continuation byte
  test_format(R"("\x{df}a")"sv,
              "{:?}",
              "\xdf"
              "a");
  test_format(R"("\x{ef}a")"sv,
              "{:?}",
              "\xef"
              "a");
  test_format(R"("\x{ef}\x{bf}a")"sv,
              "{:?}",
              "\xef\xbf"
              "a");
  test_format(R"("\x{f7}a")"sv,
              "{:?}",
              "\xf7"
              "a");
  test_format(R"("\x{f7}\x{bf}a")"sv,
              "{:?}",
              "\xf7\xbf"
              "a");
  test_format(R"("\x{f7}\x{bf}\x{bf}a")"sv,
              "{:?}",
              "\xf7\xbf\xbf"
              "a");

  test_format(R"("a\x{f1}\x{80}\x{80}\x{e1}\x{80}\x{c2}b")"sv,
              "{:?}",
              "a"
              "\xf1\x80\x80\xe1\x80\xc2"
              "b");

  // Code unit out of range
  test_format(R"("\u{10ffff}")"sv, "{:?}", "\xf4\x8f\xbf\xbf");               // last valid code point
  test_format(R"("\x{f4}\x{90}\x{80}\x{80}")"sv, "{:?}", "\xf4\x90\x80\x80"); // first invalid code point
  test_format(R"("\x{f5}\x{b1}\x{b2}\x{b3}")"sv, "{:?}", "\xf5\xb1\xb2\xb3");
  test_format(R"("\x{f7}\x{bf}\x{bf}\x{bf}")"sv, "{:?}", "\xf7\xbf\xbf\xbf"); // largest encoded code point
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#  ifdef _LIBCPP_SHORT_WCHAR
static void test_ill_formed_utf16() {
  using namespace std::literals;

  // Too few code units
  test_format(LR"("\x{d800}")"sv, L"{:?}", L"\xd800");
  test_format(LR"("\x{dbff}")"sv, L"{:?}", L"\xdbff");

  // Start with low surrogate pair
  test_format(LR"("\x{dc00}a")"sv,
              L"{:?}",
              L"\xdc00"
              "a");
  test_format(LR"("\x{dfff}a")"sv,
              L"{:?}",
              L"\xdfff"
              "a");

  // Only high surrogate pair
  test_format(LR"("\x{d800}a")"sv,
              L"{:?}",
              L"\xd800"
              "a");
  test_format(LR"("\x{dbff}a")"sv,
              L"{:?}",
              L"\xdbff"
              "a");
}
#  else // _LIBCPP_SHORT_WCHAR
static void test_ill_formed_utf32() {
  using namespace std::literals;

  test_format(LR"("\u{10ffff}")"sv, L"{:?}", L"\x10ffff");     // last valid code point
  test_format(LR"("\x{110000}")"sv, L"{:?}", L"\x110000");     // first invalid code point
  test_format(LR"("\x{ffffffff}")"sv, L"{:?}", L"\xffffffff"); // largest encoded code point
}

#  endif // _LIBCPP_SHORT_WCHAR
#endif   // TEST_HAS_NO_WIDE_CHARACTERS

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  test_ill_formed_utf8();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#  ifdef _LIBCPP_SHORT_WCHAR
  test_ill_formed_utf16();
#  else  // _LIBCPP_SHORT_WCHAR
  test_ill_formed_utf32();
#  endif // _LIBCPP_SHORT_WCHAR
#endif   // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
