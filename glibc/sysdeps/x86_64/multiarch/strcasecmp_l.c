/* Multiple versions of strcasecmp_l.
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
# define strcasecmp_l __redirect_strcasecmp_l
# define __strcasecmp_l __redirect___strcasecmp_l
# include <string.h>
# undef strcasecmp_l
# undef __strcasecmp_l

# define SYMBOL_NAME strcasecmp_l
# include "ifunc-strcasecmp.h"

libc_ifunc_redirected (__redirect_strcasecmp_l, __strcasecmp_l,
		       IFUNC_SELECTOR ());

weak_alias (__strcasecmp_l, strcasecmp_l)
# ifdef SHARED
__hidden_ver1 (__strcasecmp_l, __GI___strcasecmp_l,
	       __redirect___strcasecmp_l)
  __attribute__ ((visibility ("hidden")));
# endif
#endif
