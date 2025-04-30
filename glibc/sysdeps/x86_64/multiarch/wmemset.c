/* Multiple versions of wmemset.
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
# define wmemset __redirect_wmemset
# define __wmemset __redirect___wmemset
# include <wchar.h>
# undef wmemset
# undef __wmemset

# define SYMBOL_NAME wmemset
# include "ifunc-wmemset.h"

libc_ifunc_redirected (__redirect_wmemset, __wmemset, IFUNC_SELECTOR ());
weak_alias (__wmemset, wmemset)

# ifdef SHARED
__hidden_ver1 (__wmemset, __GI___wmemset, __redirect___wmemset)
  __attribute__ ((visibility ("hidden")));
__hidden_ver1 (wmemset, __GI_wmemset, __redirect_wmemset)
  __attribute__ ((visibility ("hidden")));
# endif
#endif
