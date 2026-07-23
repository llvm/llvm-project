//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal implementation helper for nl_langinfo and nl_langinfo_l.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LANGINFO_LANGINFO_UTILS_H
#define LLVM_LIBC_SRC_LANGINFO_LANGINFO_UTILS_H

#include "hdr/langinfo_macros.h"
#include "hdr/types/locale_t.h"
#include "hdr/types/nl_item.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/locale/locale.h"
#include "src/locale/locale_data.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE const char *get_langinfo_string(nl_item item,
                                            [[maybe_unused]] locale_t loc) {
  const LcCtypeData *ctype_data;
  const LcNumericData *numeric_data;
  const LcTimeData *time_data;
  const LcMonetaryData *monetary_data;
  const LcMessagesData *messages_data;

  if constexpr (ENABLE_RUNTIME_LOCALE) {
    ctype_data = loc->ctype();
    numeric_data = loc->numeric();
    time_data = loc->time();
    monetary_data = loc->monetary();
    messages_data = loc->messages();
  } else {
    ctype_data = DEFAULT_LOCALE_IS_UTF8 ? &UTF8_CTYPE_DATA : &C_CTYPE_DATA;
    numeric_data = &C_NUMERIC_DATA;
    time_data = &C_TIME_DATA;
    monetary_data = &C_MONETARY_DATA;
    messages_data = &C_MESSAGES_DATA;
  }

  switch (item) {
  case CODESET:
    return ctype_data->codeset;
  case RADIXCHAR:
    return numeric_data->radixchar;
  case THOUSEP:
    return numeric_data->thousep;
  case D_T_FMT:
    return time_data->d_t_fmt;
  case D_FMT:
    return time_data->d_fmt;
  case T_FMT:
    return time_data->t_fmt;
  case T_FMT_AMPM:
    return time_data->t_fmt_ampm;
  case AM_STR:
    return time_data->am_str;
  case PM_STR:
    return time_data->pm_str;
  case DAY_1:
  case DAY_2:
  case DAY_3:
  case DAY_4:
  case DAY_5:
  case DAY_6:
  case DAY_7:
    return time_data->days[item - DAY_1];
  case ABDAY_1:
  case ABDAY_2:
  case ABDAY_3:
  case ABDAY_4:
  case ABDAY_5:
  case ABDAY_6:
  case ABDAY_7:
    return time_data->ab_days[item - ABDAY_1];
  case MON_1:
  case MON_2:
  case MON_3:
  case MON_4:
  case MON_5:
  case MON_6:
  case MON_7:
  case MON_8:
  case MON_9:
  case MON_10:
  case MON_11:
  case MON_12:
    return time_data->months[item - MON_1];
  case ABMON_1:
  case ABMON_2:
  case ABMON_3:
  case ABMON_4:
  case ABMON_5:
  case ABMON_6:
  case ABMON_7:
  case ABMON_8:
  case ABMON_9:
  case ABMON_10:
  case ABMON_11:
  case ABMON_12:
    return time_data->ab_months[item - ABMON_1];
  case ERA:
    return time_data->era;
  case ERA_D_FMT:
    return time_data->era_d_fmt;
  case ERA_D_T_FMT:
    return time_data->era_d_t_fmt;
  case ERA_T_FMT:
    return time_data->era_t_fmt;
  case ALT_DIGITS:
    return time_data->alt_digits;
  case CRNCYSTR:
    return monetary_data->crncystr;
  case YESEXPR:
    return messages_data->yesexpr;
  case NOEXPR:
    return messages_data->noexpr;
  default:
    return "";
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LANGINFO_LANGINFO_UTILS_H
