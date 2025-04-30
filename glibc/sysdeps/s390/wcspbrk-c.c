/* Default wcspbrk implementation for S/390.
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

#include <ifunc-wcspbrk.h>

#if HAVE_WCSPBRK_C
# if HAVE_WCSPBRK_IFUNC || HAVE_WCSPBRK_Z13
#  define WCSPBRK WCSPBRK_C

#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_def
#   if ! defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
#    define libc_hidden_def(name)			\
  __hidden_ver1 (__wcspbrk_c, __GI_wcspbrk, __wcspbrk_c);
#   else
#    define libc_hidden_def(name)
#   endif
#  endif
# endif

# include <wcsmbs/wcspbrk.c>
#endif
