//===-- Strftime related internals -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string_view.h"

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

static constexpr int NUM_DAYS = 7;
static constexpr int NUM_MONTHS = 12;
static constexpr int YEAR_BASE = 1900;

/* The number of days from the first day of the first ISO week of this
   year to the year day YDAY with week day WDAY.  ISO weeks start on
   Monday; the first ISO week has the year's first Thursday.  YDAY may
   be as small as YDAY_MINIMUM.  */
static constexpr int ISO_WEEK_START_WDAY = 1; /* Monday */
static constexpr int ISO_WEEK1_WDAY = 4;      /* Thursday */
static constexpr int YDAY_MINIMUM = -366;

static constexpr cpp::array<cpp::string_view, NUM_DAYS> day_names = {
    "Sunday",   "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday"};

static constexpr cpp::array<cpp::string_view, NUM_MONTHS> month_names = {
    "January", "February", "March",     "April",   "May",      "June",
    "July",    "August",   "September", "October", "November", "December"};

static constexpr cpp::array<cpp::string_view, NUM_DAYS> abbreviated_day_names =
    {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

static constexpr cpp::array<cpp::string_view, NUM_MONTHS>
    abbreviated_month_names = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL
