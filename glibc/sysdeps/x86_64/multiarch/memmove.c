/* Multiple versions of memmmove.
   All versions must be listed in ifunc-impl-list.c.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
# define memmove __redirect_memmove
# include <string.h>
# undef memmove

# define SYMBOL_NAME memmove
# include "ifunc-memmove.h"

libc_ifunc_redirected (__redirect_memmove, __libc_memmove,
		       IFUNC_SELECTOR ());

strong_alias (__libc_memmove, memmove);
# ifdef SHARED
__hidden_ver1 (__libc_memmove, __GI_memmove, __redirect_memmove)
  __attribute__ ((visibility ("hidden")));
# endif
#endif
