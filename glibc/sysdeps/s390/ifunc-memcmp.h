/* memcmp variant information on S/390 version.
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
# define HAVE_MEMCMP_IFUNC	1
#else
# define HAVE_MEMCMP_IFUNC	0
#endif

#if defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define MEMCMP_DEFAULT		MEMCMP_Z196
# define HAVE_MEMCMP_Z900_G5	0
# define HAVE_MEMCMP_Z10	0
# define HAVE_MEMCMP_Z196	1
#elif defined HAVE_S390_MIN_Z10_ZARCH_ASM_SUPPORT
# define MEMCMP_DEFAULT		MEMCMP_Z10
# define HAVE_MEMCMP_Z900_G5	0
# define HAVE_MEMCMP_Z10	1
# define HAVE_MEMCMP_Z196	HAVE_MEMCMP_IFUNC
#else
# define MEMCMP_DEFAULT		MEMCMP_Z900_G5
# define HAVE_MEMCMP_Z900_G5	1
# define HAVE_MEMCMP_Z10	HAVE_MEMCMP_IFUNC
# define HAVE_MEMCMP_Z196	HAVE_MEMCMP_IFUNC
#endif

#if HAVE_MEMCMP_Z900_G5
# define MEMCMP_Z900_G5		__memcmp_default
#else
# define MEMCMP_Z900_G5		NULL
#endif

#if HAVE_MEMCMP_Z10
# define MEMCMP_Z10		__memcmp_z10
#else
# define MEMCMP_Z10		NULL
#endif

#if HAVE_MEMCMP_Z196
# define MEMCMP_Z196		__memcmp_z196
#else
# define MEMCMP_Z196		NULL
#endif
