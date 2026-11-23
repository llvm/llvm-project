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

#ifndef LIBC_CONF_DEFAULT_LOCALE
#error "LIBC_CONF_DEFAULT_LOCALE must be defined by the build system."
#endif

LIBC_INLINE_VAR constexpr cpp::string_view DEFAULT_LOCALE_NAME =
    LIBC_CONF_DEFAULT_LOCALE;

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

#ifdef LIBC_CONF_ENABLE_RUNTIME_LOCALE
LIBC_INLINE_VAR constexpr bool ENABLE_RUNTIME_LOCALE = true;
#else
LIBC_INLINE_VAR constexpr bool ENABLE_RUNTIME_LOCALE = false;
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

template <bool EnableRuntimeLocale> struct LocaleStorage {
  const LcCtypeData *ctype_ptr;
  const LcNumericData *numeric_ptr;
  const LcTimeData *time_ptr;
  const void *collate_ptr;
  const LcMonetaryData *monetary_ptr;
  const LcMessagesData *messages_ptr;

  LIBC_INLINE constexpr LocaleStorage(const LcCtypeData *c,
                                      const LcNumericData *num,
                                      const LcTimeData *t, const void *col,
                                      const LcMonetaryData *mon,
                                      const LcMessagesData *msg)
      : ctype_ptr(c), numeric_ptr(num), time_ptr(t), collate_ptr(col),
        monetary_ptr(mon), messages_ptr(msg) {}

  LIBC_INLINE constexpr const LcCtypeData *ctype() const { return ctype_ptr; }
  LIBC_INLINE constexpr const LcNumericData *numeric() const {
    return numeric_ptr;
  }
  LIBC_INLINE constexpr const LcTimeData *time() const { return time_ptr; }
  LIBC_INLINE constexpr const LcMonetaryData *monetary() const {
    return monetary_ptr;
  }
  LIBC_INLINE constexpr const LcMessagesData *messages() const {
    return messages_ptr;
  }
};

template <> struct LocaleStorage<false> {
  LIBC_INLINE constexpr LocaleStorage(const LcCtypeData *,
                                      const LcNumericData *, const LcTimeData *,
                                      const void *, const LcMonetaryData *,
                                      const LcMessagesData *) {}

  LIBC_INLINE constexpr const LcCtypeData *ctype() const { return nullptr; }
  LIBC_INLINE constexpr const LcNumericData *numeric() const { return nullptr; }
  LIBC_INLINE constexpr const LcTimeData *time() const { return nullptr; }
  LIBC_INLINE constexpr const LcMonetaryData *monetary() const {
    return nullptr;
  }
  LIBC_INLINE constexpr const LcMessagesData *messages() const {
    return nullptr;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LOCALE_LOCALE_DATA_H
