/* Shared HTM header.  Work around false transactional execution facility
   intrinsics.

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <htmintrin.h>

#ifdef __s390x__
# define TX_FPRS_BYTES 64
# define TX_SAVE_FPRS						\
  "   std %%f8, 0(%[R_FPRS])\n\t"				\
  "   std %%f9, 8(%[R_FPRS])\n\t"				\
  "   std %%f10, 16(%[R_FPRS])\n\t"				\
  "   std %%f11, 24(%[R_FPRS])\n\t"				\
  "   std %%f12, 32(%[R_FPRS])\n\t"				\
  "   std %%f13, 40(%[R_FPRS])\n\t"				\
  "   std %%f14, 48(%[R_FPRS])\n\t"				\
  "   std %%f15, 56(%[R_FPRS])\n\t"

# define TX_RESTORE_FPRS					\
  "   ld %%f8, 0(%[R_FPRS])\n\t"				\
  "   ld %%f9, 8(%[R_FPRS])\n\t"				\
  "   ld %%f10, 16(%[R_FPRS])\n\t"				\
  "   ld %%f11, 24(%[R_FPRS])\n\t"				\
  "   ld %%f12, 32(%[R_FPRS])\n\t"				\
  "   ld %%f13, 40(%[R_FPRS])\n\t"				\
  "   ld %%f14, 48(%[R_FPRS])\n\t"				\
  "   ld %%f15, 56(%[R_FPRS])\n\t"

#else

# define TX_FPRS_BYTES 16
# define TX_SAVE_FPRS						\
  "   std %%f4, 0(%[R_FPRS])\n\t"				\
  "   std %%f6, 8(%[R_FPRS])\n\t"

# define TX_RESTORE_FPRS					\
  "   ld %%f4, 0(%[R_FPRS])\n\t"				\
  "   ld %%f6, 8(%[R_FPRS])\n\t"

#endif /* ! __s390x__  */

/* Use own inline assembly instead of __builtin_tbegin, as tbegin
   has to filter program interruptions which can't be done with the builtin.
   Now the fprs have to be saved / restored here, too.
   The fpc is also not saved / restored with the builtin.
   The used inline assembly does not clobber the volatile fprs / vrs!
   Clobbering the latter ones would force the compiler to save / restore
   the call saved fprs as those overlap with the vrs, but they only need to be
   restored if the transaction fails but not if the transaction is successfully
   started.  Thus the user of the tbegin macros in this header file has to
   compile the file / function with -msoft-float.  It prevents gcc from using
   fprs / vrs.  */
#define __libc_tbegin(tdb) __libc_tbegin_base(tdb,,,)

#define __libc_tbegin_retry_output_regs , [R_TX_CNT] "+&d" (__tx_cnt)
#define __libc_tbegin_retry_input_regs(retry_cnt) , [R_RETRY] "d" (retry_cnt)
#define __libc_tbegin_retry_abort_path_insn				\
  /* If tbegin returned _HTM_TBEGIN_TRANSIENT, retry immediately so	\
     that max tbegin_cnt transactions are tried.  Otherwise return and	\
     let the caller of this macro do the fallback path.  */		\
  "   jnh 1f\n\t" /* cc 1/3: jump to fallback path.  */			\
  /* tbegin returned _HTM_TBEGIN_TRANSIENT: retry with transaction.  */ \
  "   crje %[R_TX_CNT], %[R_RETRY], 1f\n\t" /* Reached max retries?  */	\
  "   ahi %[R_TX_CNT], 1\n\t"						\
  "   ppa %[R_TX_CNT], 0, 1\n\t" /* Transaction-Abort Assist.  */	\
  "   j 2b\n\t" /* Loop to tbegin.  */

/* Same as __libc_tbegin except if tbegin aborts with _HTM_TBEGIN_TRANSIENT.
   Then this macros restores the fpc, fprs and automatically retries up to
   retry_cnt tbegins.  Further saving of the state is omitted as it is already
   saved.  This macro calls tbegin at most as retry_cnt + 1 times.  */
#define __libc_tbegin_retry(tdb, retry_cnt)				\
  ({ int __ret;								\
    int __tx_cnt = 0;							\
    __ret = __libc_tbegin_base(tdb,					\
			       __libc_tbegin_retry_abort_path_insn,	\
			       __libc_tbegin_retry_output_regs,		\
			       __libc_tbegin_retry_input_regs(retry_cnt)); \
    __ret;								\
  })

#define __libc_tbegin_base(tdb, abort_path_insn, output_regs, input_regs) \
  ({ int __ret;								\
     int __fpc;								\
     char __fprs[TX_FPRS_BYTES];					\
     __asm__ __volatile__ (".machine push\n\t"				\
			   ".machinemode \"zarch_nohighgprs\"\n\t"	\
			   ".machine \"all\"\n\t"			\
			   /* Save state at the outermost transaction.	\
			      As extracting nesting depth is expensive	\
			      on at least zEC12, save fprs at inner	\
			      transactions, too.			\
			      The fpc and fprs are saved here as they	\
			      are not saved by tbegin.  There exist no	\
			      call-saved vrs, thus they are not saved	\
			      here.  */					\
			   "   efpc %[R_FPC]\n\t"			\
			   TX_SAVE_FPRS					\
			   /* Begin transaction: save all gprs, allow	\
			      ar modification and fp operations.  Some	\
			      program-interruptions (e.g. a null	\
			      pointer access) are filtered and the	\
			      transaction will abort.  In this case	\
			      the normal lock path will execute it	\
			      again and result in a core dump wich does	\
			      now show at tbegin but the real executed	\
			      instruction.				\
			      However it is not guaranteed that this	\
			      retry operate on the same data and thus	\
			      may not end in an program-interruption.	\
			      Note: This could also be used to probe	\
			      memory for being accessible!  */		\
			   "2: tbegin 0, 0xFF0E\n\t"			\
			   /* Branch away in abort case (this is the	\
			      prefered sequence.  See PoP in chapter 5	\
			      Transactional-Execution Facility		\
			      Operation).  */				\
			   "   jnz 0f\n\t"				\
			   /* Transaction has successfully started.  */	\
			   "   lhi %[R_RET], 0\n\t"			\
			   "   j 1f\n\t"				\
			   /* Transaction has aborted.  Now we are at	\
			      the outermost transaction.  Restore fprs	\
			      and fpc. */				\
			   "0: ipm %[R_RET]\n\t"			\
			   "   srl %[R_RET], 28\n\t"			\
			   "   sfpc %[R_FPC]\n\t"			\
			   TX_RESTORE_FPRS				\
			   abort_path_insn				\
			   "1:\n\t"					\
			   ".machine pop\n"				\
			   : [R_RET] "=&d" (__ret),			\
			     [R_FPC] "=&d" (__fpc)			\
			     output_regs				\
			   : [R_FPRS] "a" (__fprs)			\
			     input_regs					\
			   : "cc", "memory");				\
     __ret;								\
     })

/* These builtins are usable in context of glibc lock elision code without any
   changes.  Use them.  */
#define __libc_tend()							\
  ({ __asm__ __volatile__ (".machine push\n\t"				\
			   ".machinemode \"zarch_nohighgprs\"\n\t"	\
			   ".machine \"all\"\n\t");			\
    int __ret = __builtin_tend ();					\
    __asm__ __volatile__ (".machine pop");				\
    __ret;								\
  })

#define __libc_tabort(abortcode)					\
  __asm__ __volatile__ (".machine push\n\t"				\
			".machinemode \"zarch_nohighgprs\"\n\t"		\
			".machine \"all\"\n\t");			\
  __builtin_tabort (abortcode);						\
  __asm__ __volatile__ (".machine pop")

#define __libc_tx_nesting_depth() \
  ({ __asm__ __volatile__ (".machine push\n\t"				\
			   ".machinemode \"zarch_nohighgprs\"\n\t"	\
			   ".machine \"all\"\n\t");			\
    int __ret = __builtin_tx_nesting_depth ();				\
    __asm__ __volatile__ (".machine pop");				\
    __ret;								\
  })

#endif
