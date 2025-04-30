/* Default memmem implementation with vector string functions for S/390.
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

#if HAVE_MEMMEM_Z13
# include <string.h>
# if HAVE_MEMMEM_IFUNC || MEMMEM_Z13_ONLY_USED_AS_FALLBACK

#  ifndef _LIBC
#   define memmem MEMMEM_Z13
#  else
#   define __memmem MEMMEM_Z13
#  endif

#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_def
#   undef libc_hidden_weak

#   if HAVE_MEMMEM_C || MEMMEM_Z13_ONLY_USED_AS_FALLBACK
#    define libc_hidden_def(name)
#    define libc_hidden_weak(name)
#   else
#    define libc_hidden_def(name)				\
  strong_alias (__memmem_vx, __memmem_vx_1);			\
  __hidden_ver1 (__memmem_vx, __GI___memmem, __memmem_vx);

#    define libc_hidden_weak(name)					\
  __hidden_ver1 (__memmem_vx_1, __GI_memmem, __memmem_vx_1) __attribute__((weak));
#   endif
#  endif

#  undef weak_alias
#  define weak_alias(a, b)
# endif

# ifdef USE_MULTIARCH
extern __typeof (memchr) __memchr_vx attribute_hidden;
# define memchr __memchr_vx

extern __typeof (memcmp) __memcmp_z196 attribute_hidden;
# define memcmp __memcmp_z196
# endif

# include <string/memmem.c>
#endif
