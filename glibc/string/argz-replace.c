/* String replacement in an argz vector
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>

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
#include <argz.h>

/* Append BUF, of length BUF_LEN to *TO, of length *TO_LEN, reallocating and
   updating *TO & *TO_LEN appropriately.  If an allocation error occurs,
   *TO's old value is freed, and *TO is set to 0.  */
static void
str_append (char **to, size_t *to_len, const char *buf, const size_t buf_len)
{
  size_t new_len = *to_len + buf_len;
  char *new_to = realloc (*to, new_len + 1);

  if (new_to)
    {
      *((char *) __mempcpy (new_to + *to_len, buf, buf_len)) = '\0';
      *to = new_to;
      *to_len = new_len;
    }
  else
    {
      free (*to);
      *to = 0;
    }
}

/* Replace any occurrences of the string STR in ARGZ with WITH, reallocating
   ARGZ as necessary.  If REPLACE_COUNT is non-zero, *REPLACE_COUNT will be
   incremented by number of replacements performed.  */
error_t
__argz_replace (char **argz, size_t *argz_len, const char *str, const char *with,
		unsigned *replace_count)
{
  error_t err = 0;

  if (str && *str)
    {
      char *arg = 0;
      char *src = *argz;
      size_t src_len = *argz_len;
      char *dst = 0;
      size_t dst_len = 0;
      int delayed_copy = 1;	/* True while we've avoided copying anything.  */
      size_t str_len = strlen (str), with_len = strlen (with);

      while (!err && (arg = argz_next (src, src_len, arg)))
	{
	  char *match = strstr (arg, str);
	  if (match)
	    {
	      char *from = match + str_len;
	      size_t to_len = match - arg;
	      char *to = __strndup (arg, to_len);

	      while (to && from)
		{
		  str_append (&to, &to_len, with, with_len);
		  if (to)
		    {
		      match = strstr (from, str);
		      if (match)
			{
			  str_append (&to, &to_len, from, match - from);
			  from = match + str_len;
			}
		      else
			{
			  str_append (&to, &to_len, from, strlen (from));
			  from = 0;
			}
		    }
		}

	      if (to)
		{
		  if (delayed_copy)
		    /* We avoided copying SRC to DST until we found a match;
                       now that we've done so, copy everything from the start
                       of SRC.  */
		    {
		      if (arg > src)
			err = __argz_append (&dst, &dst_len, src, (arg - src));
		      delayed_copy = 0;
		    }
		  if (! err)
		    err = __argz_add (&dst, &dst_len, to);
		  free (to);
		}
	      else
		err = ENOMEM;

	      if (replace_count)
		(*replace_count)++;
	    }
	  else if (! delayed_copy)
	    err = __argz_add (&dst, &dst_len, arg);
	}

      if (! err)
	{
	  if (! delayed_copy)
	    /* We never found any instances of str.  */
	    {
	      free (src);
	      *argz = dst;
	      *argz_len = dst_len;
	    }
	}
      else if (dst_len > 0)
	free (dst);
    }

  return err;
}
weak_alias (__argz_replace, argz_replace)
