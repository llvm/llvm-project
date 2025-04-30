/* Shared HTM header.  Emulate transactional execution facility intrinsics for
   compilers and assemblers that do not support the intrinsics and instructions
   yet.

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

#ifndef _HTM_H
#define _HTM_H 1

#ifdef __ASSEMBLER__

/* tbegin.  */
.macro TBEGIN
	.long 0x7c00051d
.endm

/* tend. 0  */
.macro TEND
	.long 0x7c00055d
.endm

/* tabort. code  */
.macro TABORT code
	.byte 0x7c
	.byte \code
	.byte 0x07
	.byte 0x1d
.endm

/*"TEXASR - Transaction EXception And Summary Register"
   mfspr %dst,130  */
.macro TEXASR dst
	mfspr \dst,130
.endm

#else

#include <bits/endian.h>

/* Official HTM intrinsics interface matching GCC, but works
   on older GCC compatible compilers and binutils.
   We should somehow detect if the compiler supports it, because
   it may be able to generate slightly better code.  */

#define TBEGIN ".long 0x7c00051d"
#define TEND   ".long 0x7c00055d"
#if __BYTE_ORDER == __LITTLE_ENDIAN
# define TABORT ".byte 0x1d,0x07,%1,0x7c"
#else
# define TABORT ".byte 0x7c,%1,0x07,0x1d"
#endif

#define __force_inline        inline __attribute__((__always_inline__))

#ifndef __HTM__

#define _TEXASRU_EXTRACT_BITS(TEXASR,BITNUM,SIZE) \
  (((TEXASR) >> (31-(BITNUM))) & ((1<<(SIZE))-1))
#define _TEXASRU_FAILURE_PERSISTENT(TEXASRU) \
  _TEXASRU_EXTRACT_BITS(TEXASRU, 7, 1)

#define _tbegin()			\
  ({ unsigned int __ret;		\
     asm volatile (			\
       TBEGIN "\t\n"			\
       "mfcr   %0\t\n"			\
       "rlwinm %0,%0,3,1\t\n"		\
       "xori %0,%0,1\t\n"		\
       : "=r" (__ret) :			\
       : "cr0", "memory");		\
     __ret;				\
  })

#define _tend()				\
  ({ unsigned int __ret;		\
     asm volatile (			\
       TEND "\t\n"			\
       "mfcr   %0\t\n"			\
       "rlwinm %0,%0,3,1\t\n"		\
       "xori %0,%0,1\t\n"		\
       : "=r" (__ret) :			\
       : "cr0", "memory");		\
     __ret;				\
  })

#define _tabort(__code)			\
  ({ unsigned int __ret;		\
     asm volatile (			\
       TABORT "\t\n"			\
       "mfcr   %0\t\n"			\
       "rlwinm %0,%0,3,1\t\n"		\
       "xori %0,%0,1\t\n"		\
       : "=r" (__ret) : "r" (__code)	\
       : "cr0", "memory");		\
     __ret;				\
  })

#define _texasru()			\
  ({ unsigned long __ret;		\
     asm volatile (			\
       "mfspr %0,131\t\n"		\
       : "=r" (__ret));			\
     __ret;				\
  })

#define __libc_tbegin(tdb)       _tbegin ()
#define __libc_tend(nested)      _tend ()
#define __libc_tabort(abortcode) _tabort (abortcode)
#define __builtin_get_texasru()  _texasru ()

#else
# include <htmintrin.h>

# ifdef __TM_FENCE__
   /* New GCC behavior.  */
#  define __libc_tbegin(R)  __builtin_tbegin (R)
#  define __libc_tend(R)    __builtin_tend (R)
#  define __libc_tabort(R)  __builtin_tabort (R)
# else
   /* Workaround an old GCC behavior. Earlier releases of GCC 4.9 and 5.0,
      didn't use to treat __builtin_tbegin, __builtin_tend and
      __builtin_tabort as compiler barriers, moving instructions into and
      out the transaction.
      Remove this when glibc drops support for GCC 5.0.  */
#  define __libc_tbegin(R)			\
   ({ __asm__ volatile("" ::: "memory");	\
     unsigned int __ret = __builtin_tbegin (R);	\
     __asm__ volatile("" ::: "memory");		\
     __ret;					\
   })
#  define __libc_tabort(R)			\
  ({ __asm__ volatile("" ::: "memory");		\
    unsigned int __ret = __builtin_tabort (R);	\
    __asm__ volatile("" ::: "memory");		\
    __ret;					\
  })
#  define __libc_tend(R)			\
   ({ __asm__ volatile("" ::: "memory");	\
     unsigned int __ret = __builtin_tend (R);	\
     __asm__ volatile("" ::: "memory");		\
     __ret;					\
   })
# endif /* __TM_FENCE__  */
#endif /* __HTM__  */

#endif /* __ASSEMBLER__ */

/* Definitions used for TEXASR Failure code (bits 0:7).  If the failure
   should be persistent, the abort code must be odd.  0xd0 through 0xff
   are reserved for the kernel and potential hypervisor.  */
#define _ABORT_PERSISTENT      0x01   /* An unspecified persistent abort.  */
#define _ABORT_LOCK_BUSY       0x34   /* Busy lock, not persistent.  */
#define _ABORT_NESTED_TRYLOCK  (0x32 | _ABORT_PERSISTENT)
#define _ABORT_SYSCALL         (0x30 | _ABORT_PERSISTENT)

#endif
