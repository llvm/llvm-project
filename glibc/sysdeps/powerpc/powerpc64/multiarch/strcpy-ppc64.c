/* Multiarch strcpy for PPC64.
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

#include <string.h>

#if defined SHARED && IS_IN (libc)
extern __typeof (memcpy) __memcpy_ppc attribute_hidden;
extern __typeof (strlen) __strlen_ppc attribute_hidden;
extern __typeof (strcpy) __strcpy_ppc attribute_hidden;

# define STRCPY __strcpy_ppc
# define memcpy __memcpy_ppc
# define strlen __strlen_ppc

# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name) \
  __hidden_ver1 (__strcpy_ppc, __GI_strcpy, __strcpy_ppc);
#endif

#include <string/strcpy.c>
