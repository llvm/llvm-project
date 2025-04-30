/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1995.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include "localeinfo.h"

/* This table's entries are taken from POSIX.2 Table 2-9
   ``LC_MONETARY Category Definition in the POSIX Locale'',
   with additions from ISO 14652, section 4.4.  */
static const char not_available[] = "\377";
static const uint32_t conversion_rate[] = { 1, 1 };

const struct __locale_data _nl_C_LC_MONETARY attribute_hidden =
{
  _nl_C_name,
  NULL, 0, 0,			/* no file mapped */
  { NULL, },			/* no cached data */
  UNDELETABLE,
  0,
  46,
  {
    { .string = "" },
    { .string = "" },
    { .string = "" },
    { .string = "" },
    { .string = "" },
    { .string = "" },
    { .string = "" },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = "-" },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = "" },
    { .string = "" },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .string = not_available },
    { .word = 10101 },
    { .word = 99991231 },
    { .word = 10101 },
    { .word = 99991231 },
    { .string = (const char *) conversion_rate },
    { .word = (unsigned int) L'\0' },
    { .word = (unsigned int) L'\0' },
    { .string = _nl_C_codeset }
  }
};
