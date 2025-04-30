/* Multiple versions of vectorized sinf, vector length is 8.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#define SYMBOL_NAME _ZGVdN8v_sinf
#include "ifunc-mathvec-avx2.h"

libc_ifunc_redirected (REDIRECT_NAME, SYMBOL_NAME, IFUNC_SELECTOR ());

#ifdef SHARED
__hidden_ver1 (_ZGVdN8v_sinf, __GI__ZGVdN8v_sinf,
	       __redirect__ZGVdN8v_sinf)
  __attribute__ ((visibility ("hidden")));
#endif
