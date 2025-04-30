/* Transliteration using the locale's data.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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

#include <assert.h>
#include <dlfcn.h>
#include <search.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <libc-lock.h>
#include "gconv_int.h"
#include "../locale/localeinfo.h"


int
__gconv_transliterate (struct __gconv_step *step,
		       struct __gconv_step_data *step_data,
		       const unsigned char *inbufstart,
		       const unsigned char **inbufp,
		       const unsigned char *inbufend,
		       unsigned char **outbufstart, size_t *irreversible)
{
  /* Find out about the locale's transliteration.  */
  uint_fast32_t size;
  const uint32_t *from_idx;
  const uint32_t *from_tbl;
  const uint32_t *to_idx;
  const uint32_t *to_tbl;
  const uint32_t *winbuf;
  const uint32_t *winbufend;
  uint_fast32_t low;
  uint_fast32_t high;

  /* The input buffer.  There are actually 4-byte values.  */
  winbuf = (const uint32_t *) *inbufp;
  winbufend = (const uint32_t *) inbufend;

  __gconv_fct fct = step->__fct;
#ifdef PTR_DEMANGLE
  if (step->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif

  /* If there is no transliteration information in the locale don't do
     anything and return the error.  */
  size = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_TRANSLIT_TAB_SIZE);
  if (size == 0)
    goto no_rules;

  /* Get the rest of the values.  */
  from_idx =
    (const uint32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TRANSLIT_FROM_IDX);
  from_tbl =
    (const uint32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TRANSLIT_FROM_TBL);
  to_idx =
    (const uint32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TRANSLIT_TO_IDX);
  to_tbl =
    (const uint32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TRANSLIT_TO_TBL);

  /* Test whether there is enough input.  */
  if (winbuf + 1 > winbufend)
    return (winbuf == winbufend
	    ? __GCONV_EMPTY_INPUT : __GCONV_INCOMPLETE_INPUT);

  /* The array starting at FROM_IDX contains indices to the string table
     in FROM_TBL.  The indices are sorted wrt to the strings.  I.e., we
     are doing binary search.  */
  low = 0;
  high = size;
  while (low < high)
    {
      uint_fast32_t med = (low + high) / 2;
      uint32_t idx;
      int cnt;

      /* Compare the string at this index with the string at the current
	 position in the input buffer.  */
      idx = from_idx[med];
      cnt = 0;
      do
	{
	  if (from_tbl[idx + cnt] != winbuf[cnt])
	    /* Does not match.  */
	    break;
	  ++cnt;
	}
      while (from_tbl[idx + cnt] != L'\0' && winbuf + cnt < winbufend);

      if (cnt > 0 && from_tbl[idx + cnt] == L'\0')
	{
	  /* Found a matching input sequence.  Now try to convert the
	     possible replacements.  */
	  uint32_t idx2 = to_idx[med];

	  do
	    {
	      /* Determine length of replacement.  */
	      uint_fast32_t len = 0;
	      int res;
	      const unsigned char *toinptr;
	      unsigned char *outptr;

	      while (to_tbl[idx2 + len] != L'\0')
		++len;

	      /* Try this input text.  */
	      toinptr = (const unsigned char *) &to_tbl[idx2];
	      outptr = *outbufstart;
	      res = DL_CALL_FCT (fct,
				 (step, step_data, &toinptr,
				  (const unsigned char *) &to_tbl[idx2 + len],
				  &outptr, NULL, 0, 0));
	      if (res != __GCONV_ILLEGAL_INPUT)
		{
		  /* If the conversion succeeds we have to increment the
		     input buffer.  */
		  if (res == __GCONV_EMPTY_INPUT)
		    {
		      *inbufp += cnt * sizeof (uint32_t);
		      ++*irreversible;
		      res = __GCONV_OK;
		    }
		  /* Do not increment the output pointer if we could not
		     store the entire output. */
		  if (res != __GCONV_FULL_OUTPUT)
		    *outbufstart = outptr;

		  return res;
		}

	      /* Next replacement.  */
	      idx2 += len + 1;
	    }
	  while (to_tbl[idx2] != L'\0');

	  /* Nothing found, continue searching.  */
	}
      else if (cnt > 0)
	/* This means that the input buffer contents matches a prefix of
	   an entry.  Since we cannot match it unless we get more input,
	   we will tell the caller about it.  */
	return __GCONV_INCOMPLETE_INPUT;

      if (winbuf + cnt >= winbufend || from_tbl[idx + cnt] < winbuf[cnt])
	low = med + 1;
      else
	high = med;
    }

 no_rules:
  /* Maybe the character is supposed to be ignored.  */
  if (_NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_TRANSLIT_IGNORE_LEN) != 0)
    {
      int n = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_TRANSLIT_IGNORE_LEN);
      const uint32_t *ranges =
	(const uint32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TRANSLIT_IGNORE);
      const uint32_t wc = *(const uint32_t *) (*inbufp);
      int i;

      /* Test whether there is enough input.  */
      if (winbuf + 1 > winbufend)
	return (winbuf == winbufend
		? __GCONV_EMPTY_INPUT : __GCONV_INCOMPLETE_INPUT);

      for (i = 0; i < n; ranges += 3, ++i)
	if (ranges[0] <= wc && wc <= ranges[1]
	    && (wc - ranges[0]) % ranges[2] == 0)
	  {
	    /* Matches the range.  Ignore it.  */
	    *inbufp += 4;
	    ++*irreversible;
	    return __GCONV_OK;
	  }
	else if (wc < ranges[0])
	  /* There cannot be any other matching range since they are
             sorted.  */
	  break;
    }

  /* One last chance: use the default replacement.  */
  if (_NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_TRANSLIT_DEFAULT_MISSING_LEN) != 0)
    {
      const uint32_t *default_missing = (const uint32_t *)
	_NL_CURRENT (LC_CTYPE, _NL_CTYPE_TRANSLIT_DEFAULT_MISSING);
      const unsigned char *toinptr = (const unsigned char *) default_missing;
      uint32_t len = _NL_CURRENT_WORD (LC_CTYPE,
				       _NL_CTYPE_TRANSLIT_DEFAULT_MISSING_LEN);
      unsigned char *outptr;
      int res;

      /* Test whether there is enough input.  */
      if (winbuf + 1 > winbufend)
	return (winbuf == winbufend
		? __GCONV_EMPTY_INPUT : __GCONV_INCOMPLETE_INPUT);

      outptr = *outbufstart;
      res = DL_CALL_FCT (fct,
			 (step, step_data, &toinptr,
			  (const unsigned char *) (default_missing + len),
			  &outptr, NULL, 0, 0));

      if (res != __GCONV_ILLEGAL_INPUT)
	{
	  /* If the conversion succeeds we have to increment the
	     input buffer.  */
	  if (res == __GCONV_EMPTY_INPUT)
	    {
	      /* This worked but is not reversible.  */
	      ++*irreversible;
	      *inbufp += 4;
	      res = __GCONV_OK;
	    }
	  *outbufstart = outptr;

	  return res;
	}
    }

  /* Haven't found a match.  */
  return __GCONV_ILLEGAL_INPUT;
}
libc_hidden_def (__gconv_transliterate)
