//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal structures for locale category data.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LOCALE_LOCALE_DATA_H
#define LLVM_LIBC_SRC_LOCALE_LOCALE_DATA_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#ifdef LIBC_CONF_DEFAULT_LOCALE
LIBC_INLINE_VAR constexpr cpp::string_view DEFAULT_LOCALE_NAME =
    LIBC_CONF_DEFAULT_LOCALE;
#else
LIBC_INLINE_VAR constexpr cpp::string_view DEFAULT_LOCALE_NAME = "C.UTF-8";
#endif

LIBC_INLINE constexpr bool is_c_locale_name(cpp::string_view name) {
  return name == "C" || name == "POSIX";
}

LIBC_INLINE constexpr bool is_supported_base_locale(cpp::string_view name) {
  size_t dot = name.find_first_of('.');
  if (dot != cpp::string_view::npos)
    name = name.substr(0, dot);
  return is_c_locale_name(name) || name == "en_US";
}

LIBC_INLINE constexpr bool is_utf8_locale_name(cpp::string_view name) {
  size_t dot = name.find_first_of('.');
  if (dot != cpp::string_view::npos)
    name.remove_prefix(dot + 1);
  return name == "UTF-8" || name == "utf-8" || name == "utf8" || name == "UTF8";
}

static_assert(is_supported_base_locale(DEFAULT_LOCALE_NAME) &&
                  (is_c_locale_name(DEFAULT_LOCALE_NAME) ||
                   is_utf8_locale_name(DEFAULT_LOCALE_NAME)),
              "Unsupported LIBC_CONF_DEFAULT_LOCALE value.");

LIBC_INLINE_VAR constexpr bool DEFAULT_LOCALE_IS_UTF8 =
    is_utf8_locale_name(DEFAULT_LOCALE_NAME);

#ifdef LIBC_CONF_DISABLE_RUNTIME_LOCALE
LIBC_INLINE_VAR constexpr bool DISABLE_RUNTIME_LOCALE = true;
#else
LIBC_INLINE_VAR constexpr bool DISABLE_RUNTIME_LOCALE = false;
#endif

struct LcCtypeData {
  const char *codeset;
};

struct LcNumericData {
  const char *radixchar;
  const char *thousep;
};

struct LcTimeData {
  const char *d_t_fmt;
  const char *d_fmt;
  const char *t_fmt;
  const char *t_fmt_ampm;
  const char *am_str;
  const char *pm_str;
  const char *days[7];
  const char *ab_days[7];
  const char *months[12];
  const char *ab_months[12];
  const char *era;
  const char *era_d_fmt;
  const char *era_d_t_fmt;
  const char *era_t_fmt;
  const char *alt_digits;
};

struct LcMonetaryData {
  const char *crncystr;
};

struct LcMessagesData {
  const char *yesexpr;
  const char *noexpr;
};

LIBC_INLINE_VAR constexpr LcCtypeData C_CTYPE_DATA = {"US-ASCII"};
LIBC_INLINE_VAR constexpr LcCtypeData UTF8_CTYPE_DATA = {"UTF-8"};

LIBC_INLINE_VAR constexpr LcNumericData C_NUMERIC_DATA = {".", ""};

LIBC_INLINE_VAR constexpr LcTimeData C_TIME_DATA = {
    "%a %b %e %H:%M:%S %Y",
    "%m/%d/%y",
    "%H:%M:%S",
    "%I:%M:%S %p",
    "AM",
    "PM",
    {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
     "Saturday"},
    {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"},
    {"January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"},
    {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"},
    "",
    "",
    "",
    "",
    ""};

LIBC_INLINE_VAR constexpr LcMonetaryData C_MONETARY_DATA = {""};

LIBC_INLINE_VAR constexpr LcMessagesData C_MESSAGES_DATA = {"^[yY]", "^[nN]"};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LOCALE_LOCALE_DATA_H
