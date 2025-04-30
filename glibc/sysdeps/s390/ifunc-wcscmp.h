/* wcscmp variant information on S/390 version.
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

#if defined USE_MULTIARCH && IS_IN (libc)		\
  && ! defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
# define HAVE_WCSCMP_IFUNC	1
#else
# define HAVE_WCSCMP_IFUNC	0
#endif

#ifdef HAVE_S390_VX_ASM_SUPPORT
# define HAVE_WCSCMP_IFUNC_AND_VX_SUPPORT HAVE_WCSCMP_IFUNC
#else
# define HAVE_WCSCMP_IFUNC_AND_VX_SUPPORT 0
#endif

#if defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
# define WCSCMP_DEFAULT		WCSCMP_Z13
# define HAVE_WCSCMP_C		0
# define HAVE_WCSCMP_Z13	1
#else
# define WCSCMP_DEFAULT		WCSCMP_C
# define HAVE_WCSCMP_C		1
# define HAVE_WCSCMP_Z13	HAVE_WCSCMP_IFUNC_AND_VX_SUPPORT
#endif

#if HAVE_WCSCMP_C
# define WCSCMP_C		__wcscmp_c
#else
# define WCSCMP_C		NULL
#endif

#if HAVE_WCSCMP_Z13
# define WCSCMP_Z13		__wcscmp_vx
#else
# define WCSCMP_Z13		NULL
#endif
