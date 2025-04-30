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

#include <stdlib.h>
#include <string.h>
#include <limits.h>

int
__seed48_r (unsigned short int seed16v[3], struct drand48_data *buffer)
{
  /* Save old value at a private place to be used as return value.  */
  memcpy (buffer->__old_x, buffer->__x, sizeof (buffer->__x));

  /* Install new state.  */
  buffer->__x[2] = seed16v[2];
  buffer->__x[1] = seed16v[1];
  buffer->__x[0] = seed16v[0];
  buffer->__a = 0x5deece66dull;
  buffer->__c = 0xb;
  buffer->__init = 1;

  return 0;
}
weak_alias (__seed48_r, seed48_r)
