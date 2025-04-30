/* Helper functions used by strftime/strptime to handle alternate digits.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include "../locale/localeinfo.h"
#include <libc-lock.h>
#include <stdlib.h>
#include <wchar.h>
#include <string.h>
#include <stdint.h>

/* Some of the functions here must not be used while setlocale is called.  */
__libc_rwlock_define (extern, __libc_setlocale_lock attribute_hidden)

#define CURRENT(item) (current->values[_NL_ITEM_INDEX (item)].string)
#define CURRENT_WSTR(item) \
  ((wchar_t *) current->values[_NL_ITEM_INDEX (item)].wstr)

static void
_nl_init_alt_digit (struct __locale_data *current)
{
  struct lc_time_data *data;

  if (current->private.time == NULL)
    {
      current->private.time = malloc (sizeof *current->private.time);
      if (current->private.time == NULL)
	return;
      memset (current->private.time, 0, sizeof *current->private.time);
      current->private.cleanup = &_nl_cleanup_time;
    }
  data = current->private.time;

  if (! data->alt_digits_initialized)
    {
      const char *ptr = CURRENT (ALT_DIGITS);
      size_t cnt;

      data->alt_digits_initialized = 1;

      if (ptr != NULL)
	{
	  data->alt_digits = malloc (100 * sizeof (const char *));
	  if (data->alt_digits != NULL)
	    for (cnt = 0; cnt < 100; ++cnt)
	      {
		data->alt_digits[cnt] = ptr;

		/* Skip digit format. */
		ptr = strchr (ptr, '\0') + 1;
	      }
	}
    }

}

const char *
_nl_get_alt_digit (unsigned int number, struct __locale_data *current)
{
  const char *result;

  if (number >= 100 || CURRENT (ALT_DIGITS)[0] == '\0')
    return NULL;

  __libc_rwlock_wrlock (__libc_setlocale_lock);

  if (current->private.time == NULL
      || ! current->private.time->alt_digits_initialized)
    _nl_init_alt_digit (current);

  result = ((current->private.time != NULL
	     && current->private.time->alt_digits != NULL)
	    ? current->private.time->alt_digits[number]
	    : NULL);

  __libc_rwlock_unlock (__libc_setlocale_lock);

  return result;
}


const wchar_t *
_nl_get_walt_digit (unsigned int number, struct __locale_data *current)
{
  const wchar_t *result = NULL;
  struct lc_time_data *data;

  if (number >= 100 || CURRENT_WSTR (_NL_WALT_DIGITS)[0] == L'\0')
    return NULL;

  __libc_rwlock_wrlock (__libc_setlocale_lock);

  if (current->private.time == NULL)
    {
      current->private.time = malloc (sizeof *current->private.time);
      if (current->private.time == NULL)
	goto out;
      memset (current->private.time, 0, sizeof *current->private.time);
      current->private.cleanup = &_nl_cleanup_time;
    }
  data = current->private.time;

  if (! data->walt_digits_initialized)
    {
      const wchar_t *ptr = CURRENT_WSTR (_NL_WALT_DIGITS);
      size_t cnt;

      data->walt_digits_initialized = 1;

      if (ptr != NULL)
	{
	  data->walt_digits = malloc (100 * sizeof (const uint32_t *));
	  if (data->walt_digits != NULL)
	    for (cnt = 0; cnt < 100; ++cnt)
	      {
		data->walt_digits[cnt] = ptr;

		/* Skip digit format. */
		ptr = __wcschr (ptr, L'\0') + 1;
	      }
	}
    }

  if (data->walt_digits != NULL)
    result = data->walt_digits[number];

 out:
  __libc_rwlock_unlock (__libc_setlocale_lock);

  return (wchar_t *) result;
}


int
_nl_parse_alt_digit (const char **strp, struct __locale_data *current)
{
  const char *str = *strp;
  int result = -1;
  size_t cnt;
  size_t maxlen = 0;

  if (CURRENT_WSTR (_NL_WALT_DIGITS)[0] == L'\0')
    return result;

  __libc_rwlock_wrlock (__libc_setlocale_lock);

  if (current->private.time == NULL
      || ! current->private.time->alt_digits_initialized)
    _nl_init_alt_digit (current);

  if (current->private.time != NULL
      && current->private.time->alt_digits != NULL)
    /* Matching is not unambiguous.  The alternative digits could be like
       I, II, III, ... and the first one is a substring of the second
       and third.  Therefore we must keep on searching until we found
       the longest possible match.  Note that this is not specified in
       the standard.  */
    for (cnt = 0; cnt < 100; ++cnt)
      {
	const char *const dig = current->private.time->alt_digits[cnt];
	size_t len = strlen (dig);

	if (len > maxlen && strncmp (dig, str, len) == 0)
	  {
	    maxlen = len;
	    result = (int) cnt;
	  }
      }

  __libc_rwlock_unlock (__libc_setlocale_lock);

  if (result != -1)
    *strp += maxlen;

  return result;
}
