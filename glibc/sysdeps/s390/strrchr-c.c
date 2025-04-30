/* Default strrchr implementation for S/390.
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

#include <ifunc-strrchr.h>

#if HAVE_STRRCHR_C
# if HAVE_STRRCHR_IFUNC
#  define STRRCHR STRRCHR_C
#  undef weak_alias
#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_builtin_def
#   define libc_hidden_builtin_def(name)			\
     __hidden_ver1 (__strrchr_c, __GI_strrchr, __strrchr_c);
#  endif
# endif

# include <string/strrchr.c>
#endif
