//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO PRINT Investigate see https://reviews.llvm.org/D156585
// UNSUPPORTED: no-filesystem

// XFAIL: availability-fp_to_chars-missing

// <ostream>

// void vprint_nonunicode(ostream& os, string_view fmt, format_args args);

// [ostream.formatted.print]/3
//   If the function is vprint_unicode and os is a stream that refers to
//   a terminal capable of displaying Unicode which is determined in an
//   implementation-defined manner, writes out to the terminal using the
//   native Unicode API;
// This is tested in
// test/libcxx/input.output/iostream.format/output.streams/ostream.formatted/ostream.formatted.print/vprint_unicode.pass.cpp

#include <cassert>
#include <ostream>
#include <sstream>

#include "assert_macros.h"
#include "concat_macros.h"
#include "print_tests.h"
#include "test_format_string.h"
#include "test_macros.h"

auto test_file = []<class... Args>(std::string_view expected, test_format_string<char, Args...> fmt, Args&&... args) {
  std::stringstream sstr;
  std::vprint_nonunicode(sstr, fmt.get(), std::make_format_args(args...));

  std::string out = sstr.str();
  TEST_REQUIRE(out == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
};

auto test_exception = []< class... Args>([[maybe_unused]] std::string_view what,
                                         [[maybe_unused]] std::string_view fmt,
                                         [[maybe_unused]] Args&&... args) {
  TEST_VALIDATE_EXCEPTION(
      std::format_error,
      [&]([[maybe_unused]] const std::format_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED(
                "\nFormat string   ", fmt, "\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      [&] {
        std::stringstream sstr;
        std::vprint_nonunicode(sstr, fmt, std::make_format_args(args...));
      }());
};

// [ostream.formatted.print]/3.2
//   ...
//   After constructing a sentry object, the function initializes an automatic variable via
//     string out = vformat(os.getloc(), fmt, args);
// This means if both
// - creating a sentry fails
// - the formatting fails
// the first one "wins" and the format_error is not thrown.
static void test_sentry_failure() {
  // In order for the creation of a sentry to fail a tied stream's
  // sync operation should fail.
  struct sync_failure : public std::basic_streambuf<char> {
  protected:
    int virtual sync() { return -1; }
  };
  sync_failure buf_tied;
  std::ostream os_tied(&buf_tied);
  os_tied.exceptions(std::stringstream::failbit | std::stringstream::badbit | std::stringstream::eofbit);

  std::stringstream os;
  os.tie(&os_tied);
  os.exceptions(std::stringstream::failbit | std::stringstream::badbit | std::stringstream::eofbit);

  TEST_THROWS_TYPE(std::ios_base::failure, std::vprint_nonunicode(os, "valid", std::make_format_args()));
  os_tied.clear();
  [[maybe_unused]] int arg = -10;
  TEST_THROWS_TYPE(std::ios_base::failure,
                   std::vprint_nonunicode(os, "throws exception at run-time {0:{0}}", std::make_format_args(arg)));

  os.exceptions(std::stringstream::goodbit);
  os.setstate(std::stringstream::failbit);
  std::vprint_nonunicode(
      os, "not called when the os.good() is false, so no exception is thrown {0:{0}}", std::make_format_args(arg));
}

// [ostream.formatted.print]/3.2
//   any exception thrown by the call to vformat is propagated without
//   regard to the value of os.exceptions() and without turning on
//   ios_base​::​badbit in the error state of os.
// Most invalid format strings are checked at compile-time. An invalid
// value for the width can only be tested run-time.
static void test_format_exception() {
  std::stringstream sstr;
  assert(sstr.good());

  [[maybe_unused]] int arg = -10;
  TEST_THROWS_TYPE(std::format_error, std::vprint_nonunicode(sstr, "no output {0:{0}}", std::make_format_args(arg)));
  assert(sstr.good());
  assert(sstr.str().empty());

  sstr.exceptions(std::stringstream::goodbit);
  TEST_THROWS_TYPE(std::format_error, std::vprint_nonunicode(sstr, "no output {0:{0}}", std::make_format_args(arg)));
  assert(sstr.good());
  assert(sstr.str().empty());

  sstr.exceptions(std::stringstream::failbit | std::stringstream::badbit | std::stringstream::eofbit);
  TEST_THROWS_TYPE(std::format_error, std::vprint_nonunicode(sstr, "no output {0:{0}}", std::make_format_args(arg)));
  assert(sstr.good());
  assert(sstr.str().empty());
}

static void test_write_failure() {
  // Stream that fails to write a single character.
  struct overflow_failure : public std::basic_streambuf<char> {
  protected:
    int virtual overflow(int) { return std::char_traits<char>::eof(); }
  };
  overflow_failure buf;
  std::ostream os(&buf);
  [[maybe_unused]] int arg = -10;
  os.exceptions(std::stringstream::failbit | std::stringstream::badbit | std::stringstream::eofbit);

  TEST_THROWS_TYPE(std::ios_base::failure, std::vprint_nonunicode(os, "valid", std::make_format_args()));
  os.clear();
  // When the parser would directly write to the output instead of
  // formatting first it would fail writing the first character 't' of
  // the string and result in a std::ios_base::failure exception.
  TEST_THROWS_TYPE(std::format_error,
                   std::vprint_nonunicode(os, "throws exception at run-time {0:{0}}", std::make_format_args(arg)));

  os.exceptions(std::stringstream::goodbit);
  os.clear();
  std::vprint_nonunicode(os, "valid", std::make_format_args());
  assert(os.fail());
}

static void test_stream_formatting() {
  std::stringstream sstr;
  auto test = [&]<class... Args>(std::string_view expected, test_format_string<char, Args...> fmt, Args&&... args) {
    sstr.str("");
    std::vprint_nonunicode(sstr, fmt.get(), std::make_format_args(args...));

    std::string out = sstr.str();
    TEST_REQUIRE(out == expected,
                 TEST_WRITE_CONCATENATED(
                     "\nFormat string   ", fmt.get(), "\nExpected output ", expected, "\nActual output   ", out, '\n'));
  };

  test("hello", "{}", "hello");

  sstr.width(10);
  test("     hello", "{}", "hello");

  sstr.fill('+');

  sstr.width(10);
  test("+++++hello", "{}", "hello");

  // *** Test embedded NUL character ***
  using namespace std::literals;
  sstr.width(15);
  test("++++hello\0world"sv, "hello{}{}", '\0', "world");

  // *** Test Unicode ***
  // Streams count code units not code points
  // 2-byte code points
  sstr.width(5);
  test("+++\u00a1", "{}", "\u00a1"); // INVERTED EXCLAMATION MARK
  sstr.width(5);
  test("+++\u07ff", "{}", "\u07ff"); // NKO TAMAN SIGN

  // 3-byte code points
  sstr.width(5);
  test("++\u0800", "{}", "\u0800"); // SAMARITAN LETTER ALAF
  sstr.width(5);
  test("++\ufffd", "{}", "\ufffd"); // REPLACEMENT CHARACTER

  // 4-byte code points
  sstr.width(5);
  test("+\U00010000", "{}", "\U00010000"); // LINEAR B SYLLABLE B008 A
  sstr.width(5);
  test("+\U0010FFFF", "{}", "\U0010FFFF"); // Undefined Character
}

int main(int, char**) {
  print_tests(test_file, test_exception);

  test_sentry_failure();
  test_format_exception();
  test_write_failure();
  test_stream_formatting();

  return 0;
}
