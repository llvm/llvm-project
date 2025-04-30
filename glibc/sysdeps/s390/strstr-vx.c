/* Default strstr implementation with vector string functions for S/390.
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

#if HAVE_STRSTR_Z13
# if HAVE_STRSTR_IFUNC || STRSTR_Z13_ONLY_USED_AS_FALLBACK
#  define STRSTR STRSTR_Z13
#  if defined SHARED && IS_IN (libc)
#   undef libc_hidden_builtin_def
#   if HAVE_STRSTR_C || STRSTR_Z13_ONLY_USED_AS_FALLBACK
#    define libc_hidden_builtin_def(name)
#   else
#    define libc_hidden_builtin_def(name)		\
  __hidden_ver1 (__strstr_vx, __GI_strstr, __strstr_vx);
#   endif
#  endif
# endif

# include <string.h>

# ifdef USE_MULTIARCH
extern __typeof (strchr) __strchr_vx attribute_hidden;
#  define strchr __strchr_vx

extern __typeof (strlen) __strlen_vx attribute_hidden;
#  define strlen __strlen_vx

extern __typeof (__strnlen) __strnlen_vx attribute_hidden;
#  define __strnlen __strnlen_vx

extern __typeof (memcmp) __memcmp_z196 attribute_hidden;
#  define memcmp __memcmp_z196
# endif

# include <string/strstr.c>
#endif
