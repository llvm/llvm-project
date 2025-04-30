/* Handling of dynamic sring tokens.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include "trusted-dirs.h"

#ifdef SHARED
# define IS_RTLD(l) (l) == &GL(dl_rtld_map)
#else
# define IS_RTLD(l) 0
#endif
/* Guess from the number of DSTs the length of the result string.  */
#define DL_DST_REQUIRED(l, name, len, cnt) \
  ({									      \
    size_t __len = (len);						      \
    size_t __cnt = (cnt);						      \
									      \
    if (__cnt > 0)							      \
      {									      \
	size_t dst_len;							      \
	/* Now we make a guess how many extra characters on top of the	      \
	   length of S we need to represent the result.  We know that	      \
	   we have CNT replacements.  Each at most can use		      \
	     MAX (MAX (strlen (ORIGIN), strlen (_dl_platform)),		      \
		  strlen (DL_DST_LIB))					      \
	   minus 4 (which is the length of "$LIB").			      \
									      \
	   First get the origin string if it is not available yet.	      \
	   This can only happen for the map of the executable or, when	      \
	   auditing, in ld.so.  */					      \
	if ((l)->l_origin == NULL)					      \
	  {								      \
	    assert ((l)->l_name[0] == '\0' || IS_RTLD (l));		      \
	    (l)->l_origin = _dl_get_origin ();				      \
	    dst_len = ((l)->l_origin && (l)->l_origin != (char *) -1	      \
			  ? strlen ((l)->l_origin) : 0);		      \
	  }								      \
	else								      \
	  dst_len = (l)->l_origin == (char *) -1			      \
	    ? 0 : strlen ((l)->l_origin);				      \
	dst_len = MAX (MAX (dst_len, GLRO(dl_platformlen)),		      \
		       strlen (DL_DST_LIB));				      \
	if (dst_len > 4)						      \
	  __len += __cnt * (dst_len - 4);				      \
      }									      \
									      \
    __len; })
