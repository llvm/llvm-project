//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of static locale instances and locale category data.
///
//===----------------------------------------------------------------------===//

#include "src/locale/locale.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/locale/locale_data.h"

namespace LIBC_NAMESPACE_DECL {

const LcCtypeData C_CTYPE_DATA = {"US-ASCII"};
const LcCtypeData UTF8_CTYPE_DATA = {"UTF-8"};

const LcNumericData C_NUMERIC_DATA = {".", ""};

const LcTimeData C_TIME_DATA = {
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

const LcMonetaryData C_MONETARY_DATA = {""};

const LcMessagesData C_MESSAGES_DATA = {"^[yY]", "^[nN]"};

__locale_t c_locale = {{
    reinterpret_cast<__locale_data *>(const_cast<LcCtypeData *>(&C_CTYPE_DATA)),
    reinterpret_cast<__locale_data *>(
        const_cast<LcNumericData *>(&C_NUMERIC_DATA)),
    reinterpret_cast<__locale_data *>(const_cast<LcTimeData *>(&C_TIME_DATA)),
    nullptr, // LC_COLLATE
    reinterpret_cast<__locale_data *>(
        const_cast<LcMonetaryData *>(&C_MONETARY_DATA)),
    reinterpret_cast<__locale_data *>(
        const_cast<LcMessagesData *>(&C_MESSAGES_DATA)),
}};

__locale_t utf8_locale = {{
    reinterpret_cast<__locale_data *>(
        const_cast<LcCtypeData *>(&UTF8_CTYPE_DATA)),
    reinterpret_cast<__locale_data *>(
        const_cast<LcNumericData *>(&C_NUMERIC_DATA)),
    reinterpret_cast<__locale_data *>(const_cast<LcTimeData *>(&C_TIME_DATA)),
    nullptr, // LC_COLLATE
    reinterpret_cast<__locale_data *>(
        const_cast<LcMonetaryData *>(&C_MONETARY_DATA)),
    reinterpret_cast<__locale_data *>(
        const_cast<LcMessagesData *>(&C_MESSAGES_DATA)),
}};

#ifdef LIBC_CONF_DEFAULT_LOCALE_IS_UTF8
locale_t global_locale = &utf8_locale;
#else
locale_t global_locale = &c_locale;
#endif

static LIBC_THREAD_LOCAL locale_t thread_locale = nullptr;

locale_t get_thread_locale() { return thread_locale; }

void set_thread_locale(locale_t loc) { thread_locale = loc; }

} // namespace LIBC_NAMESPACE_DECL
