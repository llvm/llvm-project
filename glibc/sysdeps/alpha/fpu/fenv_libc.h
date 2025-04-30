/* Internal libc stuff for floating point environment routines.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef _FENV_LIBC_H
#define _FENV_LIBC_H	1

#include <fenv.h>

#define FPCR_ROUND_MASK		(3UL << 58)
#define FPCR_ROUND_SHIFT	58

#define SWCR_MAP_MASK		(3UL << 12)
#define SWCR_ENABLE_SHIFT	16
#define SWCR_ENABLE_MASK	(FE_ALL_EXCEPT >> SWCR_ENABLE_SHIFT)
#define SWCR_STATUS_MASK	(FE_ALL_EXCEPT)
#define SWCR_ALL_MASK		(SWCR_ENABLE_MASK	\
				| SWCR_MAP_MASK		\
				| SWCR_STATUS_MASK)

/* These are declared for public consumption in <bits/fenv.h>.  */
libc_hidden_proto(__ieee_set_fp_control)
libc_hidden_proto(__ieee_get_fp_control)

#endif /* fenv_libc.h */
