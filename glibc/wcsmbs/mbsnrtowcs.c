/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1996.

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
#include <errno.h>
#include <gconv.h>
#include <string.h>
#include <wchar.h>
#include <wcsmbsload.h>

#include <sysdep.h>

#ifndef EILSEQ
# define EILSEQ EINVAL
#endif


/* This is the private state used if PS is NULL.  */
static mbstate_t state;

/* This is a non-standard function but it is very useful in the
   implementation of stdio because we have to deal with unterminated
   buffers.  At most NMC bytes will be converted.  */
size_t
__mbsnrtowcs (wchar_t *dst, const char **src, size_t nmc, size_t len,
	      mbstate_t *ps)
{
  const unsigned char *srcend;
  struct __gconv_step_data data;
  size_t result;
  int status;
  struct __gconv_step *towc;
  size_t dummy;
  const struct gconv_fcts *fcts;

  /* Tell where we want the result.  */
  data.__invocation_counter = 0;
  data.__internal_use = 1;
  data.__flags = __GCONV_IS_LAST;
  data.__statep = ps ?: &state;

  if (nmc == 0)
    return 0;
  srcend = (const unsigned char *) *src + __strnlen (*src, nmc - 1) + 1;

  /* Get the conversion functions.  */
  fcts = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));

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

      temp_state = *data.__statep;
      data.__statep = &temp_state;

      result = 0;
      data.__outbufend = (unsigned char *) buf + sizeof (buf);
      do
	{
	  data.__outbuf = (unsigned char *) buf;

	  status = DL_CALL_FCT (fct, (towc, &data, &inbuf, srcend, NULL,
				      &dummy, 0, 1));

	  result += (wchar_t *) data.__outbuf - buf;
	}
      while (status == __GCONV_FULL_OUTPUT);

      if ((status == __GCONV_OK || status == __GCONV_EMPTY_INPUT)
	  && ((wchar_t *) data.__outbuf)[-1] == L'\0')
	/* Don't count the NUL character in.  */
	--result;
    }
  else
    {
      /* This code is based on the safe assumption that all internal
	 multi-byte encodings use the NUL byte only to mark the end
	 of the string.  */
      data.__outbuf = (unsigned char *) dst;
      data.__outbufend = data.__outbuf + len * sizeof (wchar_t);

      status = DL_CALL_FCT (fct,
			    (towc, &data, (const unsigned char **) src, srcend,
			     NULL, &dummy, 0, 1));

      result = (wchar_t *) data.__outbuf - dst;

      /* We have to determine whether the last character converted
	 is the NUL character.  */
      if ((status == __GCONV_OK || status == __GCONV_EMPTY_INPUT)
	  && (assert (result > 0),
	      ((wchar_t *) dst)[result - 1] == L'\0'))
	{
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
weak_alias (__mbsnrtowcs, mbsnrtowcs)
