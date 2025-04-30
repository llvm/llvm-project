/* Default wmemchr implementation for S/390.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <ifunc-wmemchr.h>

#if HAVE_WMEMCHR_C
# if HAVE_WMEMCHR_IFUNC || HAVE_WMEMCHR_Z13
#  define WMEMCHR WMEMCHR_C

#  undef weak_alias
#  define weak_alias(name, alias)

#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_weak
#   define libc_hidden_weak(name)
#   undef libc_hidden_def
#   if ! defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
#    define libc_hidden_def(name)					\
  __hidden_ver1 (__wmemchr_c, __GI_wmemchr, __wmemchr_c)  __attribute__((weak)); \
  strong_alias (__wmemchr_c, __wmemchr_c_1);				\
  __hidden_ver1 (__wmemchr_c_1, __GI___wmemchr, __wmemchr_c_1);
#   else
#    define libc_hidden_def(name)
#   endif
#  endif
# endif

# include <wcsmbs/wmemchr.c>
#endif
