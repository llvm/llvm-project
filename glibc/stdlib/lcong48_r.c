/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, August 1995.

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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int
__lcong48_r (unsigned short int param[7], struct drand48_data *buffer)
{
  /* Store the given values.  */
  memcpy (buffer->__x, &param[0], sizeof (buffer->__x));
  buffer->__a = ((uint64_t) param[5] << 32 | (uint32_t) param[4] << 16
		 | param[3]);
  buffer->__c = param[6];
  buffer->__init = 1;

  return 0;
}
weak_alias (__lcong48_r, lcong48_r)
