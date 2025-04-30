/* memset variant information on S/390 version.
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

#if defined USE_MULTIARCH && IS_IN (libc)	\
  && ! defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define HAVE_MEMSET_IFUNC	1
#else
# define HAVE_MEMSET_IFUNC	0
#endif

#if defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define MEMSET_DEFAULT		MEMSET_Z196
# define BZERO_DEFAULT		BZERO_Z196
# define HAVE_MEMSET_Z900_G5	0
# define HAVE_MEMSET_Z10	0
# define HAVE_MEMSET_Z196	1
#elif defined HAVE_S390_MIN_Z10_ZARCH_ASM_SUPPORT
# define MEMSET_DEFAULT		MEMSET_Z10
# define BZERO_DEFAULT		BZERO_Z10
# define HAVE_MEMSET_Z900_G5	0
# define HAVE_MEMSET_Z10	1
# define HAVE_MEMSET_Z196	HAVE_MEMSET_IFUNC
#else
# define MEMSET_DEFAULT		MEMSET_Z900_G5
# define BZERO_DEFAULT		BZERO_Z900_G5
# define HAVE_MEMSET_Z900_G5	1
# define HAVE_MEMSET_Z10	HAVE_MEMSET_IFUNC
# define HAVE_MEMSET_Z196	HAVE_MEMSET_IFUNC
#endif

#if HAVE_MEMSET_Z10 || HAVE_MEMSET_Z196
# define HAVE_MEMSET_MVCLE	1
#else
# define HAVE_MEMSET_MVCLE	0
#endif

#if HAVE_MEMSET_Z900_G5
# define MEMSET_Z900_G5		__memset_default
# define BZERO_Z900_G5		__bzero_default
#else
# define MEMSET_Z900_G5		NULL
# define BZERO_Z900_G5		NULL
#endif

#if HAVE_MEMSET_Z10
# define MEMSET_Z10		__memset_z10
# define BZERO_Z10		__bzero_z10
#else
# define MEMSET_Z10		NULL
# define BZERO_Z10		NULL
#endif

#if HAVE_MEMSET_Z196
# define MEMSET_Z196		__memset_z196
# define BZERO_Z196		__bzero_z196
#else
# define MEMSET_Z196		NULL
# define BZERO_Z196		NULL
#endif
