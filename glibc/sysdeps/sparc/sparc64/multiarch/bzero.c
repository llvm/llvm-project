/* Multiple versions of bzero.  SPARC64/Linux version.
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
# define bzero __redirect_bzero
# include <string.h>
# undef bzero

# include <sparc-ifunc.h>

# define SYMBOL_NAME bzero
# include "ifunc-memset.h"

sparc_libc_ifunc_redirected (__redirect_bzero, __bzero, IFUNC_SELECTOR)
weak_alias (__bzero, bzero)

#endif
