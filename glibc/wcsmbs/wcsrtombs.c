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
#include <stdlib.h>
#include <gconv.h>
#include <wchar.h>
#include <wcsmbsload.h>

#include <sysdep.h>

#ifndef EILSEQ
# define EILSEQ EINVAL
#endif


/* This is the private state used if PS is NULL.  */
static mbstate_t state;

size_t
__wcsrtombs (char *dst, const wchar_t **src, size_t len, mbstate_t *ps)
{
  struct __gconv_step_data data;
  int status;
  size_t result;
  struct __gconv_step *tomb;
  const struct gconv_fcts *fcts;

  /* Tell where we want the result.  */
  data.__invocation_counter = 0;
  data.__internal_use = 1;
  data.__flags = __GCONV_IS_LAST;
  data.__statep = ps ?: &state;

  /* Get the conversion functions.  */
  fcts = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));

  /* Get the structure with the function pointers.  */
  tomb = fcts->tomb;
  __gconv_fct fct = tomb->__fct;
#ifdef PTR_DEMANGLE
  if (tomb->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif

  /* We have to handle DST == NULL special.  */
  if (dst == NULL)
    {
      mbstate_t temp_state;
      unsigned char buf[256];		/* Just an arbitrary value.  */
      const wchar_t *srcend = *src + __wcslen (*src) + 1;
      const unsigned char *inbuf = (const unsigned char *) *src;
      size_t dummy;

      temp_state = *data.__statep;
      data.__statep = &temp_state;

      result = 0;
      data.__outbufend = buf + sizeof (buf);

      do
	{
	  data.__outbuf = buf;

	  status = DL_CALL_FCT (fct, (tomb, &data, &inbuf,
				      (const unsigned char *) srcend, NULL,
				      &dummy, 0, 1));

	  /* Count the number of bytes.  */
	  result += data.__outbuf - buf;
	}
      while (status == __GCONV_FULL_OUTPUT);

      if (status == __GCONV_OK || status == __GCONV_EMPTY_INPUT)
	{
	  /* There better should be a NUL byte at the end.  */
	  assert (data.__outbuf[-1] == '\0');
	  /* Don't count the NUL character in.  */
	  --result;
	}
    }
  else
    {
      /* This code is based on the safe assumption that all internal
	 multi-byte encodings use the NUL byte only to mark the end
	 of the string.  */
      const wchar_t *srcend = *src + __wcsnlen (*src, len) + 1;
      size_t dummy;

      data.__outbuf = (unsigned char *) dst;
      data.__outbufend = (unsigned char *) dst + len;

      status = DL_CALL_FCT (fct, (tomb, &data, (const unsigned char **) src,
				  (const unsigned char *) srcend, NULL,
				  &dummy, 0, 1));

      /* Count the number of bytes.  */
      result = data.__outbuf - (unsigned char *) dst;

      /* We have to determine whether the last character converted
	 is the NUL character.  */
      if ((status == __GCONV_OK || status == __GCONV_EMPTY_INPUT)
	  && data.__outbuf[-1] == '\0')
	{
	  assert (data.__outbuf != (unsigned char *) dst);
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
      && status != __GCONV_EMPTY_INPUT)
    {
      result = (size_t) -1;
      __set_errno (EILSEQ);
    }

  return result;
}
weak_alias (__wcsrtombs, wcsrtombs)
