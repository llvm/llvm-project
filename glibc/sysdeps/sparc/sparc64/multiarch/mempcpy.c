/* Multiple versions of mempcpy.  SPARC64/Linux version.
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
# define mempcpy __redirect_mempcpy
# define __mempcpy __redirect___mempcpy
# define NO_MEMPCPY_STPCPY_REDIRECT
# define __NO_STRING_INLINES
# include <string.h>
# undef mempcpy
# undef __mempcpy

# include <sparc-ifunc.h>

# define SYMBOL_NAME mempcpy
# include "ifunc-memcpy.h"

sparc_libc_ifunc_redirected (__redirect_mempcpy, __mempcpy, IFUNC_SELECTOR)

sparc_ifunc_redirected_hidden_def (__redirect___mempcpy, __mempcpy)
weak_alias (__mempcpy, mempcpy)
sparc_ifunc_redirected_hidden_def (__redirect_mempcpy, mempcpy)
#endif
