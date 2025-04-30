/* Default strnlen implementation for PowerPC32.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#define STRNLEN  __strnlen_ppc
#ifdef SHARED
# undef libc_hidden_def
# define libc_hidden_def(name)  \
    __hidden_ver1 (__strnlen_ppc, __GI_strnlen, __strnlen_ppc); \
    strong_alias (__strnlen_ppc, __strnlen_ppc_1); \
    __hidden_ver1 (__strnlen_ppc_1, __GI___strnlen, __strnlen_ppc_1);
#endif

#include <string/strnlen.c>
