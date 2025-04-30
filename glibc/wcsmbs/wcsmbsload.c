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

#include <ctype.h>
#include <langinfo.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <locale/localeinfo.h>
#include <wcsmbsload.h>
#include <libc-lock.h>


/* These are the descriptions for the default conversion functions.  */
static const struct __gconv_step to_wc =
{
  .__shlib_handle = NULL,
  .__modname = NULL,
  .__counter = INT_MAX,
  .__from_name = (char *) "ANSI_X3.4-1968//TRANSLIT",
  .__to_name = (char *) "INTERNAL",
  .__fct = __gconv_transform_ascii_internal,
  .__btowc_fct = __gconv_btwoc_ascii,
  .__init_fct = NULL,
  .__end_fct = NULL,
  .__min_needed_from = 1,
  .__max_needed_from = 1,
  .__min_needed_to = 4,
  .__max_needed_to = 4,
  .__stateful = 0,
  .__data = NULL
};

static const struct __gconv_step to_mb =
{
  .__shlib_handle = NULL,
  .__modname = NULL,
  .__counter = INT_MAX,
  .__from_name = (char *) "INTERNAL",
  .__to_name = (char *) "ANSI_X3.4-1968//TRANSLIT",
  .__fct = __gconv_transform_internal_ascii,
  .__btowc_fct = NULL,
  .__init_fct = NULL,
  .__end_fct = NULL,
  .__min_needed_from = 4,
  .__max_needed_from = 4,
  .__min_needed_to = 1,
  .__max_needed_to = 1,
  .__stateful = 0,
  .__data = NULL
};


/* For the default locale we only have to handle ANSI_X3.4-1968.  */
const struct gconv_fcts __wcsmbs_gconv_fcts_c =
{
  .towc = (struct __gconv_step *) &to_wc,
  .towc_nsteps = 1,
  .tomb = (struct __gconv_step *) &to_mb,
  .tomb_nsteps = 1,
};


attribute_hidden
struct __gconv_step *
__wcsmbs_getfct (const char *to, const char *from, size_t *nstepsp)
{
  size_t nsteps;
  struct __gconv_step *result;
#if 0
  size_t nstateful;
  size_t cnt;
#endif

  if (__gconv_find_transform (to, from, &result, &nsteps, 0) != __GCONV_OK)
    /* Loading the conversion step is not possible.  */
    return NULL;

  /* Maybe it is someday necessary to allow more than one step.
     Currently this is not the case since the conversions handled here
     are from and to INTERNAL and there always is a converted for
     that.  It the directly following code is enabled the libio
     functions will have to allocate appropriate __gconv_step_data
     elements instead of only one.  */
#if 0
  /* Count the number of stateful conversions.  Since we will only
     have one 'mbstate_t' object available we can only deal with one
     stateful conversion.  */
  nstateful = 0;
  for (cnt = 0; cnt < nsteps; ++cnt)
    if (result[cnt].__stateful)
      ++nstateful;
  if (nstateful > 1)
#else
  if (nsteps > 1)
#endif
    {
      /* We cannot handle this case.  */
      __gconv_close_transform (result, nsteps);
      result = NULL;
    }
  else
    *nstepsp = nsteps;

  return result;
}


/* Extract from the given locale name the character set portion.  Since
   only the XPG form of the name includes this information we don't have
   to take care for the CEN form.  */
#define extract_charset_name(str) \
  ({									      \
    const char *cp = str;						      \
    char *result = NULL;						      \
									      \
    cp += strcspn (cp, "@.+,");						      \
    if (*cp == '.')							      \
      {									      \
	const char *endp = ++cp;					      \
	while (*endp != '\0' && *endp != '@')				      \
	  ++endp;							      \
	if (endp != cp)							      \
	  result = strndupa (cp, endp - cp);				      \
      }									      \
    result;								      \
  })


/* Some of the functions here must not be used while setlocale is called.  */
__libc_rwlock_define (extern, __libc_setlocale_lock attribute_hidden)

/* Load conversion functions for the currently selected locale.  */
void
__wcsmbs_load_conv (struct __locale_data *new_category)
{
  /* Acquire the lock.  */
  __libc_rwlock_wrlock (__libc_setlocale_lock);

  /* We should repeat the test since while we waited some other thread
     might have run this function.  */
  if (__glibc_likely (new_category->private.ctype == NULL))
    {
      /* We must find the real functions.  */
      const char *charset_name;
      const char *complete_name;
      struct gconv_fcts *new_fcts;
      int use_translit;

      /* Allocate the gconv_fcts structure.  */
      new_fcts = calloc (1, sizeof *new_fcts);
      if (new_fcts == NULL)
	goto failed;

      /* Get name of charset of the locale.  */
      charset_name = new_category->values[_NL_ITEM_INDEX(CODESET)].string;

      /* Does the user want transliteration?  */
      use_translit = new_category->use_translit;

      /* Normalize the name and add the slashes necessary for a
	 complete lookup.  */
      complete_name = norm_add_slashes (charset_name,
					use_translit ? "TRANSLIT" : "");

      /* It is not necessary to use transliteration in this direction
	 since the internal character set is supposed to be able to
	 represent all others.  */
      new_fcts->towc = __wcsmbs_getfct ("INTERNAL", complete_name,
					&new_fcts->towc_nsteps);
      if (new_fcts->towc != NULL)
	new_fcts->tomb = __wcsmbs_getfct (complete_name, "INTERNAL",
					  &new_fcts->tomb_nsteps);

      /* If any of the conversion functions is not available we don't
	 use any since this would mean we cannot convert back and
	 forth.  NB: NEW_FCTS was allocated with calloc.  */
      if (new_fcts->tomb == NULL)
	{
	  if (new_fcts->towc != NULL)
	    __gconv_close_transform (new_fcts->towc, new_fcts->towc_nsteps);

	  free (new_fcts);

	failed:
	  new_category->private.ctype = &__wcsmbs_gconv_fcts_c;
	}
      else
	{
	  new_category->private.ctype = new_fcts;
	  new_category->private.cleanup = &_nl_cleanup_ctype;
	}
    }

  __libc_rwlock_unlock (__libc_setlocale_lock);
}


/* Clone the current conversion function set.  */
void
__wcsmbs_clone_conv (struct gconv_fcts *copy)
{
  const struct gconv_fcts *orig;

  orig = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));

  /* Copy the data.  */
  *copy = *orig;

  /* Now increment the usage counters.  Note: This assumes
     copy->*_nsteps == 1.  The current locale holds a reference, so it
     is still there after acquiring the lock.  */

  __libc_lock_lock (__gconv_lock);

  bool overflow = false;
  if (copy->towc->__shlib_handle != NULL)
    overflow |= __builtin_add_overflow (copy->towc->__counter, 1,
					&copy->towc->__counter);
  if (copy->tomb->__shlib_handle != NULL)
    overflow |= __builtin_add_overflow (copy->tomb->__counter, 1,
					&copy->tomb->__counter);

  __libc_lock_unlock (__gconv_lock);

  if (overflow)
    __libc_fatal ("\
Fatal glibc error: gconv module reference counter overflow\n");
}


/* Get converters for named charset.  */
int
__wcsmbs_named_conv (struct gconv_fcts *copy, const char *name)
{
  copy->towc = __wcsmbs_getfct ("INTERNAL", name, &copy->towc_nsteps);
  if (copy->towc == NULL)
    return 1;

  copy->tomb = __wcsmbs_getfct (name, "INTERNAL", &copy->tomb_nsteps);
  if (copy->tomb == NULL)
    {
      __gconv_close_transform (copy->towc, copy->towc_nsteps);
      return 1;
    }

  return 0;
}

void
_nl_cleanup_ctype (struct __locale_data *locale)
{
  const struct gconv_fcts *const data = locale->private.ctype;
  if (data != NULL)
    {
      locale->private.ctype = NULL;
      locale->private.cleanup = NULL;

      /* Free the old conversions.  */
      __gconv_close_transform (data->tomb, data->tomb_nsteps);
      __gconv_close_transform (data->towc, data->towc_nsteps);
      free ((char *) data);
    }
}
