/* Table for builtin transformation mapping.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <endian.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include <gconv_int.h>

#include <assert.h>


static const struct builtin_map
{
  const char *name;
  __gconv_fct fct;
  __gconv_btowc_fct btowc_fct;

  int8_t min_needed_from;
  int8_t max_needed_from;
  int8_t min_needed_to;
  int8_t max_needed_to;

} map[] =
{
#define BUILTIN_TRANSFORMATION(From, To, Cost, Name, Fct, BtowcFct, \
			       MinF, MaxF, MinT, MaxT) \
  {									      \
    .name = Name,							      \
    .fct = Fct,								      \
    .btowc_fct = BtowcFct,						      \
									      \
    .min_needed_from = MinF,						      \
    .max_needed_from = MaxF,						      \
    .min_needed_to = MinT,						      \
    .max_needed_to = MaxT						      \
  },
#define BUILTIN_ALIAS(From, To)

#include <gconv_builtin.h>
};


void
__gconv_get_builtin_trans (const char *name, struct __gconv_step *step)
{
  size_t cnt;

  for (cnt = 0; cnt < sizeof (map) / sizeof (map[0]); ++cnt)
    if (strcmp (name, map[cnt].name) == 0)
      break;

  assert (cnt < sizeof (map) / sizeof (map[0]));

  step->__fct = map[cnt].fct;
  step->__btowc_fct = map[cnt].btowc_fct;
  step->__init_fct = NULL;
  step->__end_fct = NULL;
  step->__shlib_handle = NULL;
  step->__modname = NULL;

  step->__min_needed_from = map[cnt].min_needed_from;
  step->__max_needed_from = map[cnt].max_needed_from;
  step->__min_needed_to = map[cnt].min_needed_to;
  step->__max_needed_to = map[cnt].max_needed_to;

  /* None of the builtin converters handles stateful encoding.  */
  step->__stateful = 0;
}
