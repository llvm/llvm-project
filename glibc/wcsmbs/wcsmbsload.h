/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#ifndef _WCSMBSLOAD_H
#define _WCSMBSLOAD_H	1

#include <locale.h>
#include <wchar.h>
#include <locale/localeinfo.h>


/* Contains pointers to the used functions in the `gconv' modules.  */
struct gconv_fcts
  {
    struct __gconv_step *towc;
    size_t towc_nsteps;
    struct __gconv_step *tomb;
    size_t tomb_nsteps;
  };

/* Set of currently active conversion functions.  */
extern const struct gconv_fcts __wcsmbs_gconv_fcts_c attribute_hidden;

/* Load conversion functions for the currently selected locale.  */
extern void __wcsmbs_load_conv (struct __locale_data *new_category)
     attribute_hidden;

/* Clone the current `__wcsmbs_load_conv' value.  */
extern void __wcsmbs_clone_conv (struct gconv_fcts *copy)
     attribute_hidden;

/* Find the conversion functions for converting to and from NAME.  */
extern int __wcsmbs_named_conv (struct gconv_fcts *copy, const char *name)
     attribute_hidden;

/* Function used for the `private.cleanup' hook.  */
extern void _nl_cleanup_ctype (struct __locale_data *) attribute_hidden;


#include <iconv/gconv_int.h>


/* Load the function implementation if necessary.  */
extern struct __gconv_step *__wcsmbs_getfct (const char *to, const char *from,
					     size_t *nstepsp)
     attribute_hidden;

extern const struct __locale_data _nl_C_LC_CTYPE attribute_hidden;

/* Check whether the LC_CTYPE locale changed since the last call.
   Update the pointers appropriately.  */
static inline const struct gconv_fcts *
get_gconv_fcts (struct __locale_data *data)
{
  if (__glibc_unlikely (data->private.ctype == NULL))
    {
      if (__glibc_unlikely (data == &_nl_C_LC_CTYPE))
	return &__wcsmbs_gconv_fcts_c;
      __wcsmbs_load_conv (data);
    }
  return data->private.ctype;
}

#endif	/* wcsmbsload.h */
