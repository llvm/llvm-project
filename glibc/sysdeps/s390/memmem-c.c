/* Default memmem implementation for S/390.
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

#include <ifunc-memmem.h>

#if HAVE_MEMMEM_C
# if HAVE_MEMMEM_IFUNC
#  include <string.h>

#  ifndef _LIBC
#   define memmem MEMMEM_C
#  else
#   define __memmem MEMMEM_C
#  endif

#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_def
#   define libc_hidden_def(name)				\
  strong_alias (__memmem_c, __memmem_c_1);			\
  __hidden_ver1 (__memmem_c, __GI___memmem, __memmem_c);

#   undef libc_hidden_weak
#   define libc_hidden_weak(name)					\
  __hidden_ver1 (__memmem_c_1, __GI_memmem, __memmem_c_1) __attribute__((weak));
#  endif

#  undef weak_alias
#  define weak_alias(a, b)
# endif

# include <string/memmem.c>
#endif
