/* Duplicate handle for selection of locales.
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

#include <locale.h>
#include <libc-lock.h>
#include <stdlib.h>
#include <string.h>

#include <localeinfo.h>


/* Lock for protecting global data.  */
__libc_rwlock_define (extern , __libc_setlocale_lock attribute_hidden)


locale_t
__duplocale (locale_t dataset)
{
  /* This static object is returned for newlocale (LC_ALL_MASK, "C").  */
  if (dataset == _nl_C_locobj_ptr)
    return dataset;

  /* Handle a special value.  */
  if (dataset == LC_GLOBAL_LOCALE)
    dataset = &_nl_global_locale;

  locale_t result;
  int cnt;
  size_t names_len = 0;

  /* Calculate the total space we need to store all the names.  */
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL && dataset->__names[cnt] != _nl_C_name)
      names_len += strlen (dataset->__names[cnt]) + 1;

  /* Get memory.  */
  result = malloc (sizeof (struct __locale_struct) + names_len);

  if (result != NULL)
    {
      char *namep = (char *) (result + 1);

      /* We modify global data (the usage counts).  */
      __libc_rwlock_wrlock (__libc_setlocale_lock);

      for (cnt = 0; cnt < __LC_LAST; ++cnt)
	if (cnt != LC_ALL)
	  {
	    result->__locales[cnt] = dataset->__locales[cnt];
	    if (result->__locales[cnt]->usage_count < MAX_USAGE_COUNT)
	      ++result->__locales[cnt]->usage_count;

	    if (dataset->__names[cnt] == _nl_C_name)
	      result->__names[cnt] = _nl_C_name;
	    else
	      {
		result->__names[cnt] = namep;
		namep = __stpcpy (namep, dataset->__names[cnt]) + 1;
	      }
	  }

      /* Update the special members.  */
      result->__ctype_b = dataset->__ctype_b;
      result->__ctype_tolower = dataset->__ctype_tolower;
      result->__ctype_toupper = dataset->__ctype_toupper;

      /* It's done.  */
      __libc_rwlock_unlock (__libc_setlocale_lock);
    }

  return result;
}
weak_alias (__duplocale, duplocale)
