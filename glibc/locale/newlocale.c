/* Return a reference to locale information record.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <argz.h>
#include <libc-lock.h>
#include <errno.h>
#include <locale.h>
#include <stdlib.h>
#include <string.h>

#include "localeinfo.h"


/* Lock for protecting global data.  */
__libc_rwlock_define (extern , __libc_setlocale_lock attribute_hidden)


/* Use this when we come along an error.  */
#define ERROR_RETURN							      \
  do {									      \
    __set_errno (EINVAL);						      \
    return NULL;							      \
  } while (0)


locale_t
__newlocale (int category_mask, const char *locale, locale_t base)
{
  /* Intermediate memory for result.  */
  const char *newnames[__LC_LAST];
  struct __locale_struct result;
  locale_t result_ptr;
  char *locale_path;
  size_t locale_path_len;
  const char *locpath_var;
  int cnt;
  size_t names_len;

  /* We treat LC_ALL in the same way as if all bits were set.  */
  if (category_mask == 1 << LC_ALL)
    category_mask = (1 << __LC_LAST) - 1 - (1 << LC_ALL);

  /* Sanity check for CATEGORY argument.  */
  if ((category_mask & ~((1 << __LC_LAST) - 1 - (1 << LC_ALL))) != 0)
    ERROR_RETURN;

  /* `newlocale' does not support asking for the locale name. */
  if (locale == NULL)
    ERROR_RETURN;

  if (base == _nl_C_locobj_ptr)
    /* We're to modify BASE, returned for a previous call with "C".
       We can't really modify the read-only structure, so instead
       start over by copying it.  */
    base = NULL;

  if ((base == NULL || category_mask == (1 << __LC_LAST) - 1 - (1 << LC_ALL))
      && (category_mask == 0 || !strcmp (locale, "C")))
    /* Asking for the "C" locale needn't allocate a new object.  */
    return _nl_C_locobj_ptr;

  /* Allocate memory for the result.  */
  if (base != NULL)
    result = *base;
  else
    /* Fill with pointers to C locale data.  */
    result = _nl_C_locobj;

  /* If no category is to be set we return BASE if available or a
     dataset using the C locale data.  */
  if (category_mask == 0)
    {
      result_ptr = (locale_t) malloc (sizeof (struct __locale_struct));
      if (result_ptr == NULL)
	return NULL;
      *result_ptr = result;

      goto update;
    }

  /* We perhaps really have to load some data.  So we determine the
     path in which to look for the data now.  The environment variable
     `LOCPATH' must only be used when the binary has no SUID or SGID
     bit set.  If using the default path, we tell _nl_find_locale
     by passing null and it can check the canonical locale archive.  */
  locale_path = NULL;
  locale_path_len = 0;

  locpath_var = getenv ("LOCPATH");
  if (locpath_var != NULL && locpath_var[0] != '\0')
    {
      if (__argz_create_sep (locpath_var, ':',
			     &locale_path, &locale_path_len) != 0)
	return NULL;

      if (__argz_add_sep (&locale_path, &locale_path_len,
			  _nl_default_locale_path, ':') != 0)
	return NULL;
    }

  /* Get the names for the locales we are interested in.  We either
     allow a composite name or a single name.  */
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL)
      newnames[cnt] = locale;
  if (strchr (locale, ';') != NULL)
    {
      /* This is a composite name.  Make a copy and split it up.  */
      char *np = strdupa (locale);
      char *cp;
      int specified_mask = 0;

      while ((cp = strchr (np, '=')) != NULL)
	{
	  for (cnt = 0; cnt < __LC_LAST; ++cnt)
	    if (cnt != LC_ALL
		&& (size_t) (cp - np) == _nl_category_name_sizes[cnt]
		&& memcmp (np, (_nl_category_names_get (cnt)), cp - np) == 0)
	      break;

	  if (cnt == __LC_LAST)
	    /* Bogus category name.  */
	    ERROR_RETURN;

	  /* Found the category this clause sets.  */
	  specified_mask |= 1 << cnt;
	  newnames[cnt] = ++cp;
	  cp = strchr (cp, ';');
	  if (cp != NULL)
	    {
	      /* Examine the next clause.  */
	      *cp = '\0';
	      np = cp + 1;
	    }
	  else
	    /* This was the last clause.  We are done.  */
	    break;
	}

      if (category_mask &~ specified_mask)
	/* The composite name did not specify all categories we need.  */
	ERROR_RETURN;
    }

  /* Protect global data.  */
  __libc_rwlock_wrlock (__libc_setlocale_lock);

  /* Now process all categories we are interested in.  */
  names_len = 0;
  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    {
      if ((category_mask & 1 << cnt) != 0)
	{
	  result.__locales[cnt] = _nl_find_locale (locale_path,
						   locale_path_len,
						   cnt, &newnames[cnt]);
	  if (result.__locales[cnt] == NULL)
	    {
	    free_cnt_data_and_exit:
	      while (cnt-- > 0)
		if (((category_mask & 1 << cnt) != 0)
		    && result.__locales[cnt]->usage_count != UNDELETABLE)
		  /* We can remove the data.  */
		  _nl_remove_locale (cnt, result.__locales[cnt]);

              /* Critical section left.  */
              __libc_rwlock_unlock (__libc_setlocale_lock);
	      return NULL;
	    }

	  if (newnames[cnt] != _nl_C_name)
	    names_len += strlen (newnames[cnt]) + 1;
	}
      else if (cnt != LC_ALL && result.__names[cnt] != _nl_C_name)
	/* Tally up the unchanged names from BASE as well.  */
	names_len += strlen (result.__names[cnt]) + 1;
    }

  /* We successfully loaded all required data.  Allocate a new structure.
     We can't just reuse the BASE pointer, because the name strings are
     changing and we need the old name string area intact so we can copy
     out of it into the new one without overlap problems should some
     category's name be getting longer.  */
  result_ptr = malloc (sizeof (struct __locale_struct) + names_len);
  if (result_ptr == NULL)
    {
      cnt = __LC_LAST;
      goto free_cnt_data_and_exit;
    }

  if (base == NULL)
    {
      /* Fill in this new structure from scratch.  */

      char *namep = (char *) (result_ptr + 1);

      /* Install copied new names in the new structure's __names array.
	 If resolved to "C", that is already in RESULT.__names to start.  */
      for (cnt = 0; cnt < __LC_LAST; ++cnt)
	if ((category_mask & 1 << cnt) != 0 && newnames[cnt] != _nl_C_name)
	  {
	    result.__names[cnt] = namep;
	    namep = __stpcpy (namep, newnames[cnt]) + 1;
	  }

      *result_ptr = result;
    }
  else
    {
      /* We modify the base structure.  */

      char *namep = (char *) (result_ptr + 1);

      for (cnt = 0; cnt < __LC_LAST; ++cnt)
	if ((category_mask & 1 << cnt) != 0)
	  {
	    if (base->__locales[cnt]->usage_count != UNDELETABLE)
	      /* We can remove the old data.  */
	      _nl_remove_locale (cnt, base->__locales[cnt]);
	    result_ptr->__locales[cnt] = result.__locales[cnt];

	    if (newnames[cnt] == _nl_C_name)
	      result_ptr->__names[cnt] = _nl_C_name;
	    else
	      {
		result_ptr->__names[cnt] = namep;
		namep = __stpcpy (namep, newnames[cnt]) + 1;
	      }
	  }
	else if (cnt != LC_ALL)
	  {
	    /* The RESULT members point into the old BASE structure.  */
	    result_ptr->__locales[cnt] = result.__locales[cnt];
	    if (result.__names[cnt] == _nl_C_name)
	      result_ptr->__names[cnt] = _nl_C_name;
	    else
	      {
		result_ptr->__names[cnt] = namep;
		namep = __stpcpy (namep, result.__names[cnt]) + 1;
	      }
	  }

      free (base);
    }

  /* Critical section left.  */
  __libc_rwlock_unlock (__libc_setlocale_lock);

  /* Update the special members.  */
 update:
  {
    union locale_data_value *ctypes = result_ptr->__locales[LC_CTYPE]->values;
    result_ptr->__ctype_b = (const unsigned short int *)
      ctypes[_NL_ITEM_INDEX (_NL_CTYPE_CLASS)].string + 128;
    result_ptr->__ctype_tolower = (const int *)
      ctypes[_NL_ITEM_INDEX (_NL_CTYPE_TOLOWER)].string + 128;
    result_ptr->__ctype_toupper = (const int *)
      ctypes[_NL_ITEM_INDEX (_NL_CTYPE_TOUPPER)].string + 128;
  }

  return result_ptr;
}
weak_alias (__newlocale, newlocale)
