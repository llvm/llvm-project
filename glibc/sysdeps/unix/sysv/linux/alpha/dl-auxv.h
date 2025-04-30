/* Auxiliary vector processing for Linux/Alpha.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Scan the Aux Vector for the cache shape entries.  */

extern long __libc_alpha_cache_shape[4];

#define DL_PLATFORM_AUXV				\
      case AT_L1I_CACHESHAPE:				\
	__libc_alpha_cache_shape[0] = av->a_un.a_val;	\
	break;						\
      case AT_L1D_CACHESHAPE:				\
	__libc_alpha_cache_shape[1] = av->a_un.a_val;	\
	break;						\
      case AT_L2_CACHESHAPE:				\
	__libc_alpha_cache_shape[2] = av->a_un.a_val;	\
	break;						\
      case AT_L3_CACHESHAPE:				\
	__libc_alpha_cache_shape[3] = av->a_un.a_val;	\
	break;
