/* strstr variant information on S/390 version.
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
  && ! defined HAVE_S390_MIN_ARCH13_ZARCH_ASM_SUPPORT
# define HAVE_STRSTR_IFUNC	1
#else
# define HAVE_STRSTR_IFUNC	0
#endif

#ifdef HAVE_S390_VX_ASM_SUPPORT
# define HAVE_STRSTR_IFUNC_AND_VX_SUPPORT HAVE_STRSTR_IFUNC
#else
# define HAVE_STRSTR_IFUNC_AND_VX_SUPPORT 0
#endif

#ifdef HAVE_S390_ARCH13_ASM_SUPPORT
# define HAVE_STRSTR_IFUNC_AND_ARCH13_SUPPORT HAVE_STRSTR_IFUNC
#else
# define HAVE_STRSTR_IFUNC_AND_ARCH13_SUPPORT 0
#endif

#if defined HAVE_S390_MIN_ARCH13_ZARCH_ASM_SUPPORT
# define STRSTR_DEFAULT		STRSTR_ARCH13
# define HAVE_STRSTR_C		0
# define HAVE_STRSTR_Z13	1
# define STRSTR_Z13_ONLY_USED_AS_FALLBACK 1
# define HAVE_STRSTR_ARCH13	1
#elif defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
# define STRSTR_DEFAULT		STRSTR_Z13
# define HAVE_STRSTR_C		0
# define HAVE_STRSTR_Z13	1
# define HAVE_STRSTR_ARCH13	HAVE_STRSTR_IFUNC_AND_ARCH13_SUPPORT
#else
# define STRSTR_DEFAULT		STRSTR_C
# define HAVE_STRSTR_C		1
# define HAVE_STRSTR_Z13	HAVE_STRSTR_IFUNC_AND_VX_SUPPORT
# define HAVE_STRSTR_ARCH13	HAVE_STRSTR_IFUNC_AND_ARCH13_SUPPORT
#endif

#ifndef STRSTR_Z13_ONLY_USED_AS_FALLBACK
# define STRSTR_Z13_ONLY_USED_AS_FALLBACK 0
#endif

#if HAVE_STRSTR_C
# define STRSTR_C		__strstr_c
#else
# define STRSTR_C		NULL
#endif

#if HAVE_STRSTR_Z13
# define STRSTR_Z13		__strstr_vx
#else
# define STRSTR_Z13		NULL
#endif

#if HAVE_STRSTR_ARCH13
# define STRSTR_ARCH13		__strstr_arch13
#else
# define STRSTR_ARCH13		NULL
#endif
