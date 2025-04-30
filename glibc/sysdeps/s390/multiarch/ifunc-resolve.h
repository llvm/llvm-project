/* IFUNC resolver function for CPU specific functions.
   32/64 bit S/390 version.
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

#include <unistd.h>
#include <dl-procinfo.h>

#define S390_STFLE_BITS_Z10  34 /* General instructions extension */
#define S390_STFLE_BITS_Z196 45 /* Distinct operands, pop ... */
#define S390_STFLE_BITS_ARCH13_MIE3 61 /* Miscellaneous-Instruction-Extensions
					  Facility 3, e.g. mvcrl.  */

#define S390_IS_ARCH13_MIE3(STFLE_BITS)			\
  ((STFLE_BITS & (1ULL << (63 - S390_STFLE_BITS_ARCH13_MIE3))) != 0)

#define S390_IS_Z196(STFLE_BITS)			\
  ((STFLE_BITS & (1ULL << (63 - S390_STFLE_BITS_Z196))) != 0)

#define S390_IS_Z10(STFLE_BITS)				\
  ((STFLE_BITS & (1ULL << (63 - S390_STFLE_BITS_Z10))) != 0)

#define S390_STORE_STFLE(STFLE_BITS)					\
  /* We want just 1 double word to be returned.  */			\
  register unsigned long reg0 __asm__("0") = 0;				\
									\
  __asm__ __volatile__(".machine push"        "\n\t"			\
		       ".machine \"z9-109\""  "\n\t"			\
		       ".machinemode \"zarch_nohighgprs\"\n\t"		\
		       "stfle %0"             "\n\t"			\
		       ".machine pop"         "\n"			\
		       : "=QS" (STFLE_BITS), "+d" (reg0)		\
		       : : "cc");
#define s390_libc_ifunc_expr_stfle_init()				\
  unsigned long long stfle_bits = 0ULL;					\
  if (__glibc_likely ((hwcap & HWCAP_S390_STFLE)			\
		      && (hwcap & HWCAP_S390_ZARCH)			\
		      && (hwcap & HWCAP_S390_HIGH_GPRS)))		\
    {									\
      S390_STORE_STFLE (stfle_bits);					\
    }

#define s390_libc_ifunc_expr_init()
#define s390_libc_ifunc_expr(TYPE_FUNC, FUNC, EXPR)		\
  __ifunc (TYPE_FUNC, FUNC, EXPR, unsigned long int hwcap,	\
	   s390_libc_ifunc_expr_init);
