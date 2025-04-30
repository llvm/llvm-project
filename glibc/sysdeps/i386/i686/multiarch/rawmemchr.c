/* Multiple versions of rawmemchr.
   All versions must be listed in ifunc-impl-list.c.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Define multiple versions only for the definition in libc.  */
#if IS_IN (libc)
# define rawmemchr __redirect_rawmemchr
# define __rawmemchr __redirect___rawmemchr
# include <string.h>
# undef rawmemchr
# undef __rawmemchr

# define SYMBOL_NAME rawmemchr
# include "ifunc-sse2-bsf.h"

libc_ifunc_redirected (__redirect_rawmemchr, __rawmemchr,
		       IFUNC_SELECTOR ());

weak_alias (__rawmemchr, rawmemchr)
#endif
