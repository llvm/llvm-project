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

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

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

extern const LcCtypeData C_CTYPE_DATA;
extern const LcCtypeData UTF8_CTYPE_DATA;
extern const LcNumericData C_NUMERIC_DATA;
extern const LcTimeData C_TIME_DATA;
extern const LcMonetaryData C_MONETARY_DATA;
extern const LcMessagesData C_MESSAGES_DATA;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LOCALE_LOCALE_DATA_H
