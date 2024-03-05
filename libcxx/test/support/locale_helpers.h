//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_SUPPORT_LOCALE_HELPERS_H
#define LIBCXX_TEST_SUPPORT_LOCALE_HELPERS_H

#include <string>
#include "platform_support.h"
#include "test_macros.h"
#include "make_string.h"

#ifndef TEST_HAS_NO_WIDE_CHARACTERS

#include <cwctype>

#endif // TEST_HAS_NO_WIDE_CHARACTERS

namespace LocaleHelpers {

#ifndef TEST_HAS_NO_WIDE_CHARACTERS

std::wstring convert_thousands_sep(std::wstring const& in, wchar_t sep) {
  std::wstring out;
  bool seen_num_start = false;
  bool seen_decimal = false;
  for (unsigned i = 0; i < in.size(); ++i) {
    seen_decimal |= in[i] == L',';
    seen_num_start |= in[i] == L'-' || std::iswdigit(in[i]);
    if (seen_decimal || !seen_num_start || in[i] != L' ') {
      out.push_back(in[i]);
      continue;
    }
    assert(in[i] == L' ');
    out.push_back(sep);
  }
  return out;
}

#  if defined(_WIN32)
// This implementation is similar to the locale_guard in the private libcxx implementation headers
// but exists here for usability from the libcxx/test/std conformance test suites.
class LocaleGuard {
public:
  explicit LocaleGuard(const char* locale_in) : status_(_configthreadlocale(_ENABLE_PER_THREAD_LOCALE)) {
    assert(status_ != -1);
    // Setting the locale can be expensive even when the locale given is
    // already the current locale, so do an explicit check to see if the
    // current locale is already the one we want.
    const char* curr_locale = set_locale_asserts(nullptr);
    // If every category is the same, the locale string will simply be the
    // locale name, otherwise it will be a semicolon-separated string listing
    // each category.  In the second case, we know at least one category won't
    // be what we want, so we only have to check the first case.
    if (std::strcmp(locale_in, curr_locale) != 0) {
      locale_all_ = _strdup(curr_locale);
      assert(locale_all_ != nullptr);
      set_locale_asserts(locale_in);
    }
  }

  ~LocaleGuard() {
    // The CRT documentation doesn't explicitly say, but setlocale() does the
    // right thing when given a semicolon-separated list of locale settings
    // for the different categories in the same format as returned by
    // setlocale(LC_ALL, nullptr).
    if (locale_all_ != nullptr) {
      set_locale_asserts(locale_all_);
      free(locale_all_);
    }
    _configthreadlocale(status_);
  }

private:
  static const char* set_locale_asserts(const char* locale_in) {
    const char* new_locale = setlocale(LC_ALL, locale_in);
    assert(new_locale != nullptr);
    return new_locale;
  }

  int status_;
  char* locale_all_ = nullptr;
};

template <typename T>
std::wstring get_locale_lconv_cstr_member(const char* locale, T lconv::*cstr_member) {
  // Store and later restore current locale
  LocaleGuard g(locale);

  char* locale_set = setlocale(LC_ALL, locale);
  assert(locale_set != nullptr);
  lconv* lc            = localeconv();
  const char* selected = lc->*cstr_member;
  if (selected == nullptr) {
    // member is empty string on the locale
    return std::wstring();
  }

  std::size_t len = std::mbsrtowcs(nullptr, &selected, 0, nullptr);
  assert(len != static_cast<std::size_t>(-1));

  std::wstring ws_out(len, L'\0');
  std::mbstate_t mb = {};
  std::size_t ret   = std::mbsrtowcs(&ws_out[0], &selected, len, &mb);
  assert(ret != static_cast<std::size_t>(-1));

  return ws_out;
}

std::wstring get_locale_mon_thousands_sep(const char* locale) {
  return get_locale_lconv_cstr_member(locale, &lconv::mon_thousands_sep);
}

std::wstring get_locale_thousands_sep(const char* locale) {
  return get_locale_lconv_cstr_member(locale, &lconv::thousands_sep);
}
#  endif // _WIN32

// GLIBC 2.27 and newer use U+202F NARROW NO-BREAK SPACE as a thousands separator.
// This function converts the spaces in string inputs to U+202F if need
// be. FreeBSD's locale data also uses U+202F, since 2018.
// Windows may use U+00A0 NO-BREAK SPACE or U+0202F NARROW NO-BREAK SPACE.
std::wstring convert_mon_thousands_sep_fr_FR(std::wstring const& in) {
#  if defined(_CS_GNU_LIBC_VERSION)
  if (glibc_version_less_than("2.27"))
    return in;
  else
    return convert_thousands_sep(in, L'\u202F');
#elif defined(__FreeBSD__)
  return convert_thousands_sep(in, L'\u202F');
#elif defined(_WIN32)
  // Windows has changed it's fr thousands sep between releases,
  // so we find the host's separator instead of hard-coding it.
  std::wstring fr_sep_s = get_locale_mon_thousands_sep(LOCALE_fr_FR_UTF_8);
  assert(fr_sep_s.size() == 1);
  return convert_thousands_sep(in, fr_sep_s[0]);
#else
  return in;
#endif
}

// GLIBC 2.27 uses U+202F NARROW NO-BREAK SPACE as a thousands separator.
// FreeBSD, AIX and Windows use U+00A0 NO-BREAK SPACE.
std::wstring convert_thousands_sep_ru_RU(std::wstring const& in) {
#if defined(TEST_HAS_GLIBC)
  return convert_thousands_sep(in, L'\u202F');
#  elif defined(__FreeBSD__) || defined(_WIN32) || defined(_AIX)
  return convert_thousands_sep(in, L'\u00A0');
#  else
  return in;
#  endif
}

std::wstring negate_en_US(std::wstring s) {
#if defined(_WIN32)
  return L"(" + s + L")";
#else
  return L"-" + s;
#endif
}

#endif // TEST_HAS_NO_WIDE_CHARACTERS

std::string negate_en_US(std::string s) {
#if defined(_WIN32)
  return "(" + s + ")";
#else
  return "-" + s;
#endif
}

MultiStringType currency_symbol_ru_RU() {
#if defined(_CS_GNU_LIBC_VERSION)
  if (glibc_version_less_than("2.24"))
    return MKSTR("\u0440\u0443\u0431");
  else
    return MKSTR("\u20BD"); // U+20BD RUBLE SIGN
#elif defined(_WIN32) || defined(__FreeBSD__) || defined(_AIX)
  return MKSTR("\u20BD"); // U+20BD RUBLE SIGN
#else
  return MKSTR("\u0440\u0443\u0431.");
#endif
}

MultiStringType currency_symbol_zh_CN() {
#if defined(_WIN32)
  return MKSTR("\u00A5"); // U+00A5 YEN SIGN
#else
  return MKSTR("\uFFE5"); // U+FFE5 FULLWIDTH YEN SIGN
#endif
}

} // namespace LocaleHelpers

#endif // LIBCXX_TEST_SUPPORT_LOCALE_HELPERS_H
