/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 2002.

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
#include <ctype.h>
#include <string.h>
#include "wcsmbsload.h"
#include <dlfcn.h>
#include <errno.h>
#include <gconv.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wcsmbsload.h>

#include <sysdep.h>

#ifndef EILSEQ
# define EILSEQ EINVAL
#endif


size_t
attribute_hidden
__mbsrtowcs_l (wchar_t *dst, const char **src, size_t len, mbstate_t *ps,
	       locale_t l)
{
  struct __gconv_step_data data;
  size_t result;
  int status;
  struct __gconv_step *towc;
  size_t non_reversible;
  const struct gconv_fcts *fcts;

  /* Tell where we want the result.  */
  data.__invocation_counter = 0;
  data.__internal_use = 1;
  data.__flags = __GCONV_IS_LAST;
  data.__statep = ps;

  /* Get the conversion functions.  */
  fcts = get_gconv_fcts (l->__locales[LC_CTYPE]);

  /* Get the structure with the function pointers.  */
  towc = fcts->towc;
  __gconv_fct fct = towc->__fct;
#ifdef PTR_DEMANGLE
  if (towc->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif

  /* We have to handle DST == NULL special.  */
  if (dst == NULL)
    {
      mbstate_t temp_state;
      wchar_t buf[64];		/* Just an arbitrary size.  */
      const unsigned char *inbuf = (const unsigned char *) *src;
      const unsigned char *srcend = inbuf + strlen (*src) + 1;

      temp_state = *data.__statep;
      data.__statep = &temp_state;

      result = 0;
      data.__outbufend = (unsigned char *) buf + sizeof (buf);
      do
	{
	  data.__outbuf = (unsigned char *) buf;

	  status = DL_CALL_FCT (fct, (towc, &data, &inbuf, srcend, NULL,
				      &non_reversible, 0, 1));

	  result += (wchar_t *) data.__outbuf - buf;
	}
      while (status == __GCONV_FULL_OUTPUT);

      if (status == __GCONV_OK || status == __GCONV_EMPTY_INPUT)
	{
	  /* There better should be a NUL wide char at the end.  */
	  assert (((wchar_t *) data.__outbuf)[-1] == L'\0');
	  /* Don't count the NUL character in.  */
	  --result;
	}
    }
  else
    {
      /* This code is based on the safe assumption that all internal
	 multi-byte encodings use the NUL byte only to mark the end
	 of the string.  */
      const unsigned char *srcp = (const unsigned char *) *src;
      const unsigned char *srcend;

      data.__outbuf = (unsigned char *) dst;
      data.__outbufend = data.__outbuf + len * sizeof (wchar_t);

      status = __GCONV_FULL_OUTPUT;

      while (len > 0)
	{
	  /* Pessimistic guess as to how much input we can use.  In the
	     worst case we need one input byte for one output wchar_t.  */
	  srcend = srcp + __strnlen ((const char *) srcp, len) + 1;

	  status = DL_CALL_FCT (fct, (towc, &data, &srcp, srcend, NULL,
				      &non_reversible, 0, 1));
	  if ((status != __GCONV_EMPTY_INPUT
	       && status != __GCONV_INCOMPLETE_INPUT)
	      /* Not all input read.  */
	      || srcp != srcend
	      /* Reached the end of the input.  */
	      || srcend[-1] == '\0')
	    break;

	  len = (wchar_t *) data.__outbufend - (wchar_t *) data.__outbuf;
	}

      /* Make the end if the input known to the caller.  */
      *src = (const char *) srcp;

      result = (wchar_t *) data.__outbuf - dst;

      /* We have to determine whether the last character converted
	 is the NUL character.  */
      if ((status == __GCONV_OK || status == __GCONV_EMPTY_INPUT)
	  && ((wchar_t *) dst)[result - 1] == L'\0')
	{
	  assert (result > 0);
	  assert (__mbsinit (data.__statep));
	  *src = NULL;
	  --result;
	}
    }

  /* There must not be any problems with the conversion but illegal input
     characters.  */
  assert (status == __GCONV_OK || status == __GCONV_EMPTY_INPUT
	  || status == __GCONV_ILLEGAL_INPUT
	  || status == __GCONV_INCOMPLETE_INPUT
	  || status == __GCONV_FULL_OUTPUT);

  if (status != __GCONV_OK && status != __GCONV_FULL_OUTPUT
      && status != __GCONV_EMPTY_INPUT && status != __GCONV_INCOMPLETE_INPUT)
    {
      result = (size_t) -1;
      __set_errno (EILSEQ);
    }

  return result;
}
