/* This file is part of the GNU C Library.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.

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

#include <ldsodefs.h>

/* The code checks if _rtld_global_ro was realocated before trying to access
   the dl_hwcap field. The assembly is to make the compiler not optimize the
   test (&_rtld_global_ro != NULL), which is always true in ISO C (but not
   in that case since _rtld_global_ro might not been realocated yet).  */
#if defined(SHARED) && !IS_IN (rtld)
# define __GLRO(value) \
  ({ volatile void **__p = (volatile void**)(&_rtld_global_ro);	\
    unsigned long int __ret;					\
     asm ("# x in %0" : "+r" (__p));				\
     __ret = (__p) ? GLRO(value) : 0;				\
     __ret; })
#else
# define __GLRO(value)  GLRO(value)
#endif

/* dl_hwcap contains only the latest supported ISA, the macro checks which is
   and fills the previous ones.  */
#define INIT_ARCH() \
  unsigned long int hwcap = __GLRO(dl_hwcap); 			\
  unsigned long int __attribute__((unused)) hwcap2 = __GLRO(dl_hwcap2); \
  bool __attribute__((unused)) use_cached_memopt =		\
    __GLRO(dl_powerpc_cpu_features.use_cached_memopt);		\
  if (hwcap & PPC_FEATURE_ARCH_2_06)				\
    hwcap |= PPC_FEATURE_ARCH_2_05 |				\
	     PPC_FEATURE_POWER5_PLUS |				\
	     PPC_FEATURE_POWER5 |				\
	     PPC_FEATURE_POWER4;				\
  else if (hwcap & PPC_FEATURE_ARCH_2_05)			\
    hwcap |= PPC_FEATURE_POWER5_PLUS |				\
	     PPC_FEATURE_POWER5 |				\
	     PPC_FEATURE_POWER4;				\
  else if (hwcap & PPC_FEATURE_POWER5_PLUS)			\
    hwcap |= PPC_FEATURE_POWER5 |				\
	     PPC_FEATURE_POWER4;				\
  else if (hwcap & PPC_FEATURE_POWER5)				\
    hwcap |= PPC_FEATURE_POWER4;
