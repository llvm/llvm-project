/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1996.

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

#include <dlfcn.h>
#include <gconv.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <wcsmbsload.h>

#include <sysdep.h>


int
wctob (wint_t c)
{
  unsigned char buf[MB_LEN_MAX];
  struct __gconv_step_data data;
  wchar_t inbuf[1];
  wchar_t *inptr = inbuf;
  size_t dummy;
  int status;
  const struct gconv_fcts *fcts;

  if (c == WEOF)
    return EOF;

  /* We know that only ASCII compatible encodings are used for the
     locale and that the wide character encoding is ISO 10646.  */
  if (c >= L'\0' && c <= L'\x7f')
    return (int) c;

  /* Tell where we want the result.  */
  data.__outbuf = buf;
  data.__outbufend = buf + MB_LEN_MAX;
  data.__invocation_counter = 0;
  data.__internal_use = 1;
  data.__flags = __GCONV_IS_LAST;
  data.__statep = &data.__state;

  /* Make sure we start in the initial state.  */
  memset (&data.__state, '\0', sizeof (mbstate_t));

  /* Get the conversion functions.  */
  fcts = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));

  /* Create the input string.  */
  inbuf[0] = c;

  const unsigned char *argptr = (const unsigned char *) inptr;
  __gconv_fct fct = fcts->tomb->__fct;
#ifdef PTR_DEMANGLE
  if (fcts->tomb->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif
  status = DL_CALL_FCT (fct,
			(fcts->tomb, &data, &argptr,
			 argptr + sizeof (inbuf[0]), NULL, &dummy, 0, 1));

  /* The conversion failed or the output is too long.  */
  if ((status != __GCONV_OK && status != __GCONV_FULL_OUTPUT
       && status != __GCONV_EMPTY_INPUT)
      || data.__outbuf != (unsigned char *) (buf + 1))
    return EOF;

  return buf[0];
}
