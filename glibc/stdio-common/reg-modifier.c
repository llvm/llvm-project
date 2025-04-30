/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <limits.h>
#include <printf.h>
#include <stdlib.h>
#include <wchar.h>
#include <libc-lock.h>


struct printf_modifier_record
{
  struct printf_modifier_record *next;
  int bit;
  wchar_t str[0];
};

struct printf_modifier_record **__printf_modifier_table attribute_hidden;

__libc_lock_define_initialized (static, lock)

/* Bits to hand out.  */
static int next_bit;


int
__register_printf_modifier (const wchar_t *str)
{
  if (str[0] == L'\0')
    {
    einval:
      __set_errno (EINVAL);
      return -1;
    }

  const wchar_t *wc = str;
  while (*wc != L'\0')
    if (*wc < 0 || *wc > (wchar_t) UCHAR_MAX)
      goto einval;
    else
      ++wc;

  if (next_bit / 8 == sizeof (((struct printf_info *) NULL)->user))
    {
      __set_errno (ENOSPC);
      return -1;
    }

  int result = -1;
  __libc_lock_lock (lock);

  if (__printf_modifier_table == NULL)
    {
      __printf_modifier_table = calloc (UCHAR_MAX,
					sizeof (*__printf_modifier_table));
      if (__printf_modifier_table == NULL)
	goto out;
    }

  /* Create enough room for the string.  But we don't need the first
     character. */
  struct printf_modifier_record *newp = malloc (sizeof (*newp)
						+ ((wc - str)
						   * sizeof (wchar_t)));
  if (newp == NULL)
    goto out;

  newp->next = __printf_modifier_table[(unsigned char) *str];
  newp->bit = 1 << next_bit++;
  __wmemcpy (newp->str, str + 1, wc - str);

  __printf_modifier_table[(unsigned char) *str] = newp;

  result = newp->bit;

 out:
  __libc_lock_unlock (lock);

  return result;
}
weak_alias (__register_printf_modifier, register_printf_modifier)


#include <stdio.h>
int
attribute_hidden
__handle_registered_modifier_mb (const unsigned char **format,
				 struct printf_info *info)
{
  struct printf_modifier_record *runp = __printf_modifier_table[**format];

  int best_bit = 0;
  int best_len = 0;
  const unsigned char *best_cp = NULL;

  while (runp != NULL)
    {
      const unsigned char *cp = *format + 1;
      wchar_t *fcp = runp->str;

      while (*cp != '\0' && *fcp != L'\0')
	if (*cp != *fcp)
	  break;
	else
	  ++cp, ++fcp;

      if (*fcp == L'\0' && cp - *format > best_len)
	{
	  best_cp = cp;
	  best_len = cp - *format;
	  best_bit = runp->bit;
	}

      runp = runp->next;
    }

  if (best_bit != 0)
    {
      info->user |= best_bit;
      *format = best_cp;
      return 0;
    }

  return 1;
}


int
attribute_hidden
__handle_registered_modifier_wc (const unsigned int **format,
				 struct printf_info *info)
{
  struct printf_modifier_record *runp = __printf_modifier_table[**format];

  int best_bit = 0;
  int best_len = 0;
  const unsigned int *best_cp = NULL;

  while (runp != NULL)
    {
      const unsigned int *cp = *format + 1;
      wchar_t *fcp = runp->str;

      while (*cp != '\0' && *fcp != L'\0')
	if (*cp != *fcp)
	  break;
	else
	  ++cp, ++fcp;

      if (*fcp == L'\0' && cp - *format > best_len)
	{
	  best_cp = cp;
	  best_len = cp - *format;
	  best_bit = runp->bit;
	}

      runp = runp->next;
    }

  if (best_bit != 0)
    {
      info->user |= best_bit;
      *format = best_cp;
      return 0;
    }

  return 1;
}


libc_freeres_fn (free_mem)
{
  if (__printf_modifier_table != NULL)
    {
      for (int i = 0; i < UCHAR_MAX; ++i)
	{
	  struct printf_modifier_record *runp = __printf_modifier_table[i];
	  while (runp != NULL)
	    {
	      struct printf_modifier_record *oldp = runp;
	      runp = runp->next;
	      free (oldp);
	    }
	}
      free (__printf_modifier_table);
    }
}
