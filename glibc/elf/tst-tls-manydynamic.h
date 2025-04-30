/* Interfaces for test with many dynamic TLS variables.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#ifndef TST_TLS_MANYDYNAMIC_H
#define TST_TLS_MANYDYNAMIC_H

enum
  {
    /* This many TLS variables (and modules) are defined.  */
    COUNT = 100,

    /* Number of elements in the TLS variable.  */
    PER_VALUE_COUNT = 1,
  };

/* The TLS variables are of this type.  We use a larger type to ensure
   that we can reach the static TLS limit with COUNT variables.  */
struct value
{
  int num[PER_VALUE_COUNT];
};

/* Set the TLS variable defined in the module.  */
typedef void (*set_value_func) (const struct value *);

/* Read the TLS variable defined in the module.  */
typedef void (*get_value_func) (struct value *);

#endif /* TST_TLS_MANYDYNAMICMOD_H */
