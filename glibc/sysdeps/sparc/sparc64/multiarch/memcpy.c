/* Multiple versions of memcpy.  SPARC64/Linux version.
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

#if IS_IN (libc)
# define memcpy __redirect_memcpy
# include <string.h>
# undef memcpy

# include <sparc-ifunc.h>

# define SYMBOL_NAME memcpy
# include "ifunc-memcpy.h"

sparc_libc_ifunc_redirected (__redirect_memcpy, memcpy, IFUNC_SELECTOR)

sparc_ifunc_redirected_hidden_def (__redirect_memcpy, memcpy)
#endif
