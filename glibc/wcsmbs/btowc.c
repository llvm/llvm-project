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

#include <ctype.h>
#include <dlfcn.h>
#include <gconv.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <wcsmbsload.h>
#include <limits.h>

#include <sysdep.h>


wint_t
__btowc (int c)
{
  const struct gconv_fcts *fcts;

  /* If the parameter does not fit into one byte or it is the EOF value
     we can give the answer now.  */
  if (c < SCHAR_MIN || c > UCHAR_MAX || c == EOF)
    return WEOF;

  /* We know that only ASCII compatible encodings are used for the
     locale and that the wide character encoding is ISO 10646.  */
  if (isascii (c))
    return (wint_t) c;

  /* Get the conversion functions.  */
  fcts = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));
  __gconv_btowc_fct btowc_fct = fcts->towc->__btowc_fct;
#ifdef PTR_DEMANGLE
  if (fcts->towc->__shlib_handle != NULL)
    PTR_DEMANGLE (btowc_fct);
#endif

  if (__builtin_expect (fcts->towc_nsteps == 1, 1)
      && __builtin_expect (btowc_fct != NULL, 1))
    {
      /* Use the shortcut function.  */
      return DL_CALL_FCT (btowc_fct, (fcts->towc, (unsigned char) c));
    }
  else
    {
      /* Fall back to the slow but generic method.  */
      wchar_t result;
      struct __gconv_step_data data;
      unsigned char inbuf[1];
      const unsigned char *inptr = inbuf;
      size_t dummy;
      int status;

      /* Tell where we want the result.  */
      data.__outbuf = (unsigned char *) &result;
      data.__outbufend = data.__outbuf + sizeof (wchar_t);
      data.__invocation_counter = 0;
      data.__internal_use = 1;
      data.__flags = __GCONV_IS_LAST;
      data.__statep = &data.__state;

      /* Make sure we start in the initial state.  */
      memset (&data.__state, '\0', sizeof (mbstate_t));

      /* Create the input string.  */
      inbuf[0] = c;

      __gconv_fct fct = fcts->towc->__fct;
#ifdef PTR_DEMANGLE
      if (fcts->towc->__shlib_handle != NULL)
	PTR_DEMANGLE (fct);
#endif
      status = DL_CALL_FCT (fct, (fcts->towc, &data, &inptr, inptr + 1,
				  NULL, &dummy, 0, 1));

      if (status != __GCONV_OK && status != __GCONV_FULL_OUTPUT
	  && status != __GCONV_EMPTY_INPUT)
	/* The conversion failed.  */
	result = WEOF;

      return result;
    }
}
weak_alias (__btowc, btowc)
