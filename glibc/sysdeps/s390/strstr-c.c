/* Default strstr implementation for S/390.
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

#include <ifunc-strstr.h>

#if HAVE_STRSTR_C
# if HAVE_STRSTR_IFUNC
#  define STRSTR STRSTR_C
#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_builtin_def
#   define libc_hidden_builtin_def(name)		\
  __hidden_ver1 (__strstr_c, __GI_strstr, __strstr_c);
#  endif
# endif

# include <string/strstr.c>
#endif
