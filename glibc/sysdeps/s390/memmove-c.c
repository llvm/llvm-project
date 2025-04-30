/* Fallback C version of memmove.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <ifunc-memcpy.h>

#if HAVE_MEMMOVE_C
# if HAVE_MEMMOVE_IFUNC
/* If we use ifunc, then the memmove symbol is defined
   in sysdeps/s390/memmove.c and we use a different name here.
   Otherwise, we have to define memmove here or in
   sysdeps/s390/memcpy.S depending on the used default implementation.  */
#  define MEMMOVE MEMMOVE_C
#  if defined SHARED && IS_IN (libc)
/* Define the internal symbol.  */
#   undef libc_hidden_builtin_def
#   define libc_hidden_builtin_def(name)			\
  __hidden_ver1 (__memmove_c, __GI_memmove, __memmove_c);
#  endif
# endif

# include <string/memmove.c>
#endif
