/* memcpy variant information on S/390 version.
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

#if defined SHARED && defined USE_MULTIARCH && IS_IN (libc)	\
  && ! defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define HAVE_MEMCPY_IFUNC	1
#else
# define HAVE_MEMCPY_IFUNC	0
#endif

#if defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define MEMCPY_DEFAULT		MEMCPY_Z196
# define MEMPCPY_DEFAULT	MEMPCPY_Z196
# define HAVE_MEMCPY_Z900_G5	0
# define HAVE_MEMCPY_Z10	0
# define HAVE_MEMCPY_Z196	1
#elif defined HAVE_S390_MIN_Z10_ZARCH_ASM_SUPPORT
# define MEMCPY_DEFAULT		MEMCPY_Z10
# define MEMPCPY_DEFAULT	MEMPCPY_Z10
# define HAVE_MEMCPY_Z900_G5	0
# define HAVE_MEMCPY_Z10	1
# define HAVE_MEMCPY_Z196	HAVE_MEMCPY_IFUNC
#else
# define MEMCPY_DEFAULT		MEMCPY_Z900_G5
# define MEMPCPY_DEFAULT	MEMPCPY_Z900_G5
# define HAVE_MEMCPY_Z900_G5	1
# define HAVE_MEMCPY_Z10	HAVE_MEMCPY_IFUNC
# define HAVE_MEMCPY_Z196	HAVE_MEMCPY_IFUNC
#endif

#if defined SHARED && defined USE_MULTIARCH && IS_IN (libc)	\
  && ! defined HAVE_S390_MIN_ARCH13_ZARCH_ASM_SUPPORT
# define HAVE_MEMMOVE_IFUNC	1
#else
# define HAVE_MEMMOVE_IFUNC	0
#endif

#ifdef HAVE_S390_VX_ASM_SUPPORT
# define HAVE_MEMMOVE_IFUNC_AND_VX_SUPPORT HAVE_MEMMOVE_IFUNC
#else
# define HAVE_MEMMOVE_IFUNC_AND_VX_SUPPORT 0
#endif

#ifdef HAVE_S390_ARCH13_ASM_SUPPORT
# define HAVE_MEMMOVE_IFUNC_AND_ARCH13_SUPPORT HAVE_MEMMOVE_IFUNC
#else
# define HAVE_MEMMOVE_IFUNC_AND_ARCH13_SUPPORT 0
#endif

#if defined HAVE_S390_MIN_ARCH13_ZARCH_ASM_SUPPORT
# define MEMMOVE_DEFAULT	MEMMOVE_ARCH13
# define HAVE_MEMMOVE_C		0
# define HAVE_MEMMOVE_Z13	0
# define HAVE_MEMMOVE_ARCH13	1
#elif defined HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
# define MEMMOVE_DEFAULT	MEMMOVE_Z13
# define HAVE_MEMMOVE_C		0
# define HAVE_MEMMOVE_Z13	1
# define HAVE_MEMMOVE_ARCH13	HAVE_MEMMOVE_IFUNC_AND_ARCH13_SUPPORT
#else
# define MEMMOVE_DEFAULT	MEMMOVE_C
# define HAVE_MEMMOVE_C		1
# define HAVE_MEMMOVE_Z13	HAVE_MEMMOVE_IFUNC_AND_VX_SUPPORT
# define HAVE_MEMMOVE_ARCH13	HAVE_MEMMOVE_IFUNC_AND_ARCH13_SUPPORT
#endif

#if HAVE_MEMCPY_Z900_G5
# define MEMCPY_Z900_G5		__memcpy_default
# define MEMPCPY_Z900_G5	__mempcpy_default
#else
# define MEMCPY_Z900_G5		NULL
# define MEMPCPY_Z900_G5	NULL
#endif

#if HAVE_MEMCPY_Z10
# define MEMCPY_Z10		__memcpy_z10
# define MEMPCPY_Z10		__mempcpy_z10
#else
# define MEMCPY_Z10		NULL
# define MEMPCPY_Z10		NULL
#endif

#if HAVE_MEMCPY_Z196
# define MEMCPY_Z196		__memcpy_z196
# define MEMPCPY_Z196		__mempcpy_z196
#else
# define MEMCPY_Z196		NULL
# define MEMPCPY_Z196		NULL
#endif

#if HAVE_MEMMOVE_C
# define MEMMOVE_C		__memmove_c
#else
# define MEMMOVE_C		NULL
#endif

#if HAVE_MEMMOVE_Z13
# define MEMMOVE_Z13		__memmove_z13
#else
# define MEMMOVE_Z13		NULL
#endif

#if HAVE_MEMMOVE_ARCH13
# define MEMMOVE_ARCH13		__memmove_arch13
#else
# define MEMMOVE_ARCH13		NULL
#endif
