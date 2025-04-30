/* Definition for thread-local data handling.  NPTL/PowerPC version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#ifndef _TLS_H
#define _TLS_H	1

# include <dl-sysdep.h>

#ifndef __ASSEMBLER__
# include <stdbool.h>
# include <stddef.h>
# include <stdint.h>
# include <dl-dtv.h>

#else /* __ASSEMBLER__ */
# include <tcb-offsets.h>
# define __ASSEMBLY__
# include <asm/ptrace.h>
#endif /* __ASSEMBLER__ */

#ifndef __powerpc64__
/* Register r2 (tp) is reserved by the ABI as "thread pointer". */
# define PT_THREAD_POINTER PT_R2
# ifndef __ASSEMBLER__
register void *__thread_register __asm__ ("r2");
# endif

#else /* __powerpc64__ */
/* Register r13 (tp) is reserved by the ABI as "thread pointer". */
# define PT_THREAD_POINTER PT_R13
# ifndef __ASSEMBLER__
register void *__thread_register __asm__ ("r13");
# endif
#endif /* __powerpc64__ */

#ifndef __ASSEMBLER__

# include <hwcapinfo.h>

/* Get system call information.  */
# include <sysdep.h>

/* The TP points to the start of the thread blocks.  */
# define TLS_DTV_AT_TP	1
# define TLS_TCB_AT_TP	0

/* We use the multiple_threads field in the pthread struct */
#define TLS_MULTIPLE_THREADS_IN_TCB	1

/* Get the thread descriptor definition.  */
# include <nptl/descr.h>


/* The stack_guard is accessed directly by GCC -fstack-protector code,
   so it is a part of public ABI.  The dtv and pointer_guard fields
   are private.  */
typedef struct
{
  /* Reservation for HWCAP data.  To be accessed by GCC in
     __builtin_cpu_supports(), so it is a part of public ABI.  */
  uint64_t hwcap;
  /* Reservation for AT_PLATFORM data.  To be accessed by GCC in
     __builtin_cpu_is(), so it is a part of public ABI.  Since there
     are different ABIs for 32 and 64 bit, we put this field in a
     previously empty padding space for powerpc64.  */
#ifndef __powerpc64__
  /* Padding to maintain alignment.  */
  uint32_t padding;
  uint32_t at_platform;
#endif
  uint32_t __unused;
  /* Reservation for AT_PLATFORM data - powerpc64.  */
#ifdef __powerpc64__
  uint32_t at_platform;
#endif
  /* Reservation for Dynamic System Optimizer ABI.  */
  uintptr_t dso_slot2;
  uintptr_t dso_slot1;
  /* Reservation for tar register (ISA 2.07).  */
  uintptr_t tar_save;
  /* GCC split stack support.  */
  void *__private_ss;
  /* Reservation for the Event-Based Branching ABI.  */
  uintptr_t ebb_handler;
  uintptr_t ebb_ctx_pointer;
  uintptr_t ebb_reserved1;
  uintptr_t ebb_reserved2;
  uintptr_t pointer_guard;
  uintptr_t stack_guard;
  dtv_t *dtv;
} tcbhead_t;

/* This is the size of the initial TCB.  */
# define TLS_INIT_TCB_SIZE	0

/* Alignment requirements for the initial TCB.  */
# define TLS_INIT_TCB_ALIGN	__alignof__ (struct pthread)

/* This is the size of the TCB.  */
# define TLS_TCB_SIZE		0

/* Alignment requirements for the TCB.  */
# define TLS_TCB_ALIGN		__alignof__ (struct pthread)

/* This is the size we need before TCB.  */
# define TLS_PRE_TCB_SIZE \
  (sizeof (struct pthread)						      \
   + ((sizeof (tcbhead_t) + TLS_TCB_ALIGN - 1) & ~(TLS_TCB_ALIGN - 1)))

/* The following assumes that TP (R2 or R13) points to the end of the
   TCB + 0x7000 (per the ABI).  This implies that TCB address is
   TP - 0x7000.  As we define TLS_DTV_AT_TP we can
   assume that the pthread struct is allocated immediately ahead of the
   TCB.  This implies that the pthread_descr address is
   TP - (TLS_PRE_TCB_SIZE + 0x7000).  */
# define TLS_TCB_OFFSET	0x7000

/* Install the dtv pointer.  The pointer passed is to the element with
   index -1 which contain the length.  */
# define INSTALL_DTV(tcbp, dtvp) \
  ((tcbhead_t *) (tcbp))[-1].dtv = dtvp + 1

/* Install new dtv for current thread.  */
# define INSTALL_NEW_DTV(dtv) (THREAD_DTV() = (dtv))

/* Return dtv of given thread descriptor.  */
# define GET_DTV(tcbp)	(((tcbhead_t *) (tcbp))[-1].dtv)

/* Code to initially initialize the thread pointer.  This might need
   special attention since 'errno' is not yet available and if the
   operation can cause a failure 'errno' must not be touched.  */
# define TLS_INIT_TP(tcbp) \
  ({ 									      \
    __thread_register = (void *) (tcbp) + TLS_TCB_OFFSET;		      \
    THREAD_SET_HWCAP (__tcb_hwcap);					      \
    THREAD_SET_AT_PLATFORM (__tcb_platform);				      \
    NULL;								      \
  })

/* Value passed to 'clone' for initialization of the thread register.  */
# define TLS_DEFINE_INIT_TP(tp, pd) \
    void *tp = (void *) (pd) + TLS_TCB_OFFSET + TLS_PRE_TCB_SIZE;	      \
    (((tcbhead_t *) ((char *) tp - TLS_TCB_OFFSET))[-1].hwcap) =	      \
      THREAD_GET_HWCAP ();						      \
    (((tcbhead_t *) ((char *) tp - TLS_TCB_OFFSET))[-1].at_platform) =	      \
      THREAD_GET_AT_PLATFORM ();

/* Return the address of the dtv for the current thread.  */
# define THREAD_DTV() \
    (((tcbhead_t *) (__thread_register - TLS_TCB_OFFSET))[-1].dtv)

/* Return the thread descriptor for the current thread.  */
# define THREAD_SELF \
    ((struct pthread *) (__thread_register \
			 - TLS_TCB_OFFSET - TLS_PRE_TCB_SIZE))

/* Magic for libthread_db to know how to do THREAD_SELF.  */
# define DB_THREAD_SELF							      \
  REGISTER (32, 32, PT_THREAD_POINTER * 4,				      \
	    - TLS_TCB_OFFSET - TLS_PRE_TCB_SIZE)			      \
  REGISTER (64, 64, PT_THREAD_POINTER * 8,				      \
	    - TLS_TCB_OFFSET - TLS_PRE_TCB_SIZE)

/* Read member of the thread descriptor directly.  */
# define THREAD_GETMEM(descr, member) ((void)(descr), (THREAD_SELF)->member)

/* Same as THREAD_GETMEM, but the member offset can be non-constant.  */
# define THREAD_GETMEM_NC(descr, member, idx) \
    ((void)(descr), (THREAD_SELF)->member[idx])

/* Set member of the thread descriptor directly.  */
# define THREAD_SETMEM(descr, member, value) \
    ((void)(descr), (THREAD_SELF)->member = (value))

/* Same as THREAD_SETMEM, but the member offset can be non-constant.  */
# define THREAD_SETMEM_NC(descr, member, idx, value) \
    ((void)(descr), (THREAD_SELF)->member[idx] = (value))

/* Set the stack guard field in TCB head.  */
# define THREAD_SET_STACK_GUARD(value) \
    (((tcbhead_t *) ((char *) __thread_register				      \
		     - TLS_TCB_OFFSET))[-1].stack_guard = (value))
# define THREAD_COPY_STACK_GUARD(descr) \
    (((tcbhead_t *) ((char *) (descr)					      \
		     + TLS_PRE_TCB_SIZE))[-1].stack_guard		      \
     = ((tcbhead_t *) ((char *) __thread_register			      \
		       - TLS_TCB_OFFSET))[-1].stack_guard)

/* Set the stack guard field in TCB head.  */
# define THREAD_GET_POINTER_GUARD() \
    (((tcbhead_t *) ((char *) __thread_register				      \
		     - TLS_TCB_OFFSET))[-1].pointer_guard)
# define THREAD_SET_POINTER_GUARD(value) \
    (THREAD_GET_POINTER_GUARD () = (value))
# define THREAD_COPY_POINTER_GUARD(descr) \
    (((tcbhead_t *) ((char *) (descr)					      \
		     + TLS_PRE_TCB_SIZE))[-1].pointer_guard		      \
     = THREAD_GET_POINTER_GUARD())

/* hwcap field in TCB head.  */
# define THREAD_GET_HWCAP() \
    (((tcbhead_t *) ((char *) __thread_register				      \
		     - TLS_TCB_OFFSET))[-1].hwcap)
# define THREAD_SET_HWCAP(value) \
    (THREAD_GET_HWCAP () = (value))

/* at_platform field in TCB head.  */
# define THREAD_GET_AT_PLATFORM() \
    (((tcbhead_t *) ((char *) __thread_register				      \
		     - TLS_TCB_OFFSET))[-1].at_platform)
# define THREAD_SET_AT_PLATFORM(value) \
    (THREAD_GET_AT_PLATFORM () = (value))

/* l_tls_offset == 0 is perfectly valid on PPC, so we have to use some
   different value to mean unset l_tls_offset.  */
# define NO_TLS_OFFSET		-1

/* Get and set the global scope generation counter in struct pthread.  */
#define THREAD_GSCOPE_IN_TCB      1
#define THREAD_GSCOPE_FLAG_UNUSED 0
#define THREAD_GSCOPE_FLAG_USED   1
#define THREAD_GSCOPE_FLAG_WAIT   2
#define THREAD_GSCOPE_RESET_FLAG() \
  do									     \
    { int __res								     \
	= atomic_exchange_rel (&THREAD_SELF->header.gscope_flag,	     \
			       THREAD_GSCOPE_FLAG_UNUSED);		     \
      if (__res == THREAD_GSCOPE_FLAG_WAIT)				     \
	lll_futex_wake (&THREAD_SELF->header.gscope_flag, 1, LLL_PRIVATE);   \
    }									     \
  while (0)
#define THREAD_GSCOPE_SET_FLAG() \
  do									     \
    {									     \
      THREAD_SELF->header.gscope_flag = THREAD_GSCOPE_FLAG_USED;	     \
      atomic_write_barrier ();						     \
    }									     \
  while (0)

#endif /* __ASSEMBLER__ */

#endif	/* tls.h */
