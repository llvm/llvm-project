/* Definition for thread-local data handling.  nptl/i386 version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <dl-sysdep.h>
#ifndef __ASSEMBLER__
# include <stdbool.h>
# include <stddef.h>
# include <stdint.h>
# include <stdlib.h>
# include <sysdep.h>
# include <libc-pointer-arith.h> /* For cast_to_integer. */
# include <kernel-features.h>
# include <dl-dtv.h>

typedef struct
{
  void *tcb;		/* Pointer to the TCB.  Not necessarily the
			   thread descriptor used by libpthread.  */
  dtv_t *dtv;
  void *self;		/* Pointer to the thread descriptor.  */
  int multiple_threads;
  uintptr_t sysinfo;
  uintptr_t stack_guard;
  uintptr_t pointer_guard;
  int gscope_flag;
  /* Bit 0: X86_FEATURE_1_IBT.
     Bit 1: X86_FEATURE_1_SHSTK.
   */
  unsigned int feature_1;
  /* Reservation of some values for the TM ABI.  */
  void *__private_tm[3];
  /* GCC split stack support.  */
  void *__private_ss;
  /* The lowest address of shadow stack,  */
  unsigned long ssp_base;
} tcbhead_t;

/* morestack.S in libgcc uses offset 0x30 to access __private_ss,   */
_Static_assert (offsetof (tcbhead_t, __private_ss) == 0x30,
		"offset of __private_ss != 0x30");

# define TLS_MULTIPLE_THREADS_IN_TCB 1

#else /* __ASSEMBLER__ */
# include <tcb-offsets.h>
#endif


/* Alignment requirement for the stack.  For IA-32 this is governed by
   the SSE memory functions.  */
#define STACK_ALIGN	16

#ifndef __ASSEMBLER__
/* Get system call information.  */
# include <sysdep.h>

/* The old way: using LDT.  */

/* Structure passed to `modify_ldt', 'set_thread_area', and 'clone' calls.  */
struct user_desc
{
  unsigned int entry_number;
  unsigned long int base_addr;
  unsigned int limit;
  unsigned int seg_32bit:1;
  unsigned int contents:2;
  unsigned int read_exec_only:1;
  unsigned int limit_in_pages:1;
  unsigned int seg_not_present:1;
  unsigned int useable:1;
  unsigned int empty:25;
};

/* Initializing bit fields is slow.  We speed it up by using a union.  */
union user_desc_init
{
  struct user_desc desc;
  unsigned int vals[4];
};


/* This is the size of the initial TCB.  Can't be just sizeof (tcbhead_t),
   because NPTL getpid, __libc_alloca_cutoff etc. need (almost) the whole
   struct pthread even when not linked with -lpthread.  */
# define TLS_INIT_TCB_SIZE sizeof (struct pthread)

/* Alignment requirements for the initial TCB.  */
# define TLS_INIT_TCB_ALIGN __alignof__ (struct pthread)

/* This is the size of the TCB.  */
# define TLS_TCB_SIZE sizeof (struct pthread)

/* Alignment requirements for the TCB.  */
# define TLS_TCB_ALIGN __alignof__ (struct pthread)

/* The TCB can have any size and the memory following the address the
   thread pointer points to is unspecified.  Allocate the TCB there.  */
# define TLS_TCB_AT_TP	1
# define TLS_DTV_AT_TP	0

/* Get the thread descriptor definition.  */
# include <nptl/descr.h>


/* Install the dtv pointer.  The pointer passed is to the element with
   index -1 which contain the length.  */
# define INSTALL_DTV(descr, dtvp) \
  ((tcbhead_t *) (descr))->dtv = (dtvp) + 1

/* Install new dtv for current thread.  */
# define INSTALL_NEW_DTV(dtvp) \
  ({ struct pthread *__pd;						      \
     THREAD_SETMEM (__pd, header.dtv, (dtvp)); })

/* Return dtv of given thread descriptor.  */
# define GET_DTV(descr) \
  (((tcbhead_t *) (descr))->dtv)

/* Macros to load from and store into segment registers.  */
# ifndef TLS_GET_GS
#  define TLS_GET_GS() \
  ({ int __seg; __asm ("movw %%gs, %w0" : "=q" (__seg)); __seg & 0xffff; })
# endif
# ifndef TLS_SET_GS
#  define TLS_SET_GS(val) \
  __asm ("movw %w0, %%gs" :: "q" (val))
# endif

#ifdef NEED_DL_SYSINFO
# define INIT_SYSINFO \
  _head->sysinfo = GLRO(dl_sysinfo)
# define SETUP_THREAD_SYSINFO(pd) \
  ((pd)->header.sysinfo = THREAD_GETMEM (THREAD_SELF, header.sysinfo))
# define CHECK_THREAD_SYSINFO(pd) \
  assert ((pd)->header.sysinfo == THREAD_GETMEM (THREAD_SELF, header.sysinfo))
#else
# define INIT_SYSINFO
#endif

#define LOCK_PREFIX "lock;"

static inline void __attribute__ ((unused, always_inline))
tls_fill_user_desc (union user_desc_init *desc,
                    unsigned int entry_number,
                    void *pd)
{
  desc->vals[0] = entry_number;
  /* The 'base_addr' field.  Pointer to the TCB.  */
  desc->vals[1] = (unsigned long int) pd;
  /* The 'limit' field.  We use 4GB which is 0xfffff pages.  */
  desc->vals[2] = 0xfffff;
  /* Collapsed value of the bitfield:
     .seg_32bit = 1
     .contents = 0
     .read_exec_only = 0
     .limit_in_pages = 1
     .seg_not_present = 0
     .useable = 1 */
  desc->vals[3] = 0x51;
}

/* Code to initially initialize the thread pointer.  This might need
   special attention since 'errno' is not yet available and if the
   operation can cause a failure 'errno' must not be touched.  */
# define TLS_INIT_TP(thrdescr) \
  ({ void *_thrdescr = (thrdescr);					      \
     tcbhead_t *_head = _thrdescr;					      \
     union user_desc_init _segdescr;					      \
     int _result;							      \
									      \
     _head->tcb = _thrdescr;						      \
     /* For now the thread descriptor is at the same address.  */	      \
     _head->self = _thrdescr;						      \
     /* New syscall handling support.  */				      \
     INIT_SYSINFO;							      \
									      \
     /* Let the kernel pick a value for the 'entry_number' field.  */	      \
     tls_fill_user_desc (&_segdescr, -1, _thrdescr);			      \
									      \
     /* Install the TLS.  */						      \
     _result = INTERNAL_SYSCALL_CALL (set_thread_area, &_segdescr.desc);      \
									      \
     if (_result == 0)							      \
       /* We know the index in the GDT, now load the segment register.	      \
	  The use of the GDT is described by the value 3 in the lower	      \
	  three bits of the segment descriptor value.			      \
									      \
	  Note that we have to do this even if the numeric value of	      \
	  the descriptor does not change.  Loading the segment register	      \
	  causes the segment information from the GDT to be loaded	      \
	  which is necessary since we have changed it.   */		      \
       TLS_SET_GS (_segdescr.desc.entry_number * 8 + 3);		      \
									      \
     _result == 0 ? NULL						      \
     : "set_thread_area failed when setting up thread-local storage\n"; })

# define TLS_DEFINE_INIT_TP(tp, pd)					      \
  union user_desc_init _segdescr;					      \
  /* Find the 'entry_number' field that the kernel selected in TLS_INIT_TP.   \
     The first three bits of the segment register value select the GDT,	      \
     ignore them.  We get the index from the value of the %gs register in     \
     the current thread.  */						      \
  tls_fill_user_desc (&_segdescr, TLS_GET_GS () >> 3, pd);		      \
  const struct user_desc *tp = &_segdescr.desc


/* Return the address of the dtv for the current thread.  */
# define THREAD_DTV() \
  ({ struct pthread *__pd;						      \
     THREAD_GETMEM (__pd, header.dtv); })


/* Return the thread descriptor for the current thread.

   The contained asm must *not* be marked volatile since otherwise
   assignments like
	pthread_descr self = thread_self();
   do not get optimized away.  */
# if __GNUC_PREREQ (6, 0)
#  define THREAD_SELF \
  (*(struct pthread *__seg_gs *) offsetof (struct pthread, header.self))
# else
#  define THREAD_SELF \
  ({ struct pthread *__self;						      \
     asm ("movl %%gs:%c1,%0" : "=r" (__self)				      \
	  : "i" (offsetof (struct pthread, header.self)));		      \
     __self;})
# endif

/* Magic for libthread_db to know how to do THREAD_SELF.  */
# define DB_THREAD_SELF \
  REGISTER_THREAD_AREA (32, offsetof (struct user_regs_struct, xgs), 3) \
  REGISTER_THREAD_AREA (64, 26 * 8, 3) /* x86-64's user_regs_struct->gs */


/* Read member of the thread descriptor directly.  */
# define THREAD_GETMEM(descr, member) \
  ({ __typeof (descr->member) __value;					      \
     _Static_assert (sizeof (__value) == 1				      \
		     || sizeof (__value) == 4				      \
		     || sizeof (__value) == 8,				      \
		     "size of per-thread data");			      \
     if (sizeof (__value) == 1)						      \
       asm volatile ("movb %%gs:%P2,%b0"				      \
		     : "=q" (__value)					      \
		     : "0" (0), "i" (offsetof (struct pthread, member)));     \
     else if (sizeof (__value) == 4)					      \
       asm volatile ("movl %%gs:%P1,%0"					      \
		     : "=r" (__value)					      \
		     : "i" (offsetof (struct pthread, member)));	      \
     else /* 8 */								      \
       {								      \
	 asm volatile ("movl %%gs:%P1,%%eax\n\t"			      \
		       "movl %%gs:%P2,%%edx"				      \
		       : "=A" (__value)					      \
		       : "i" (offsetof (struct pthread, member)),	      \
			 "i" (offsetof (struct pthread, member) + 4));	      \
       }								      \
     __value; })


/* Same as THREAD_GETMEM, but the member offset can be non-constant.  */
# define THREAD_GETMEM_NC(descr, member, idx) \
  ({ __typeof (descr->member[0]) __value;				      \
     _Static_assert (sizeof (__value) == 1				      \
		     || sizeof (__value) == 4				      \
		     || sizeof (__value) == 8,				      \
		     "size of per-thread data");			      \
     if (sizeof (__value) == 1)						      \
       asm volatile ("movb %%gs:%P2(%3),%b0"				      \
		     : "=q" (__value)					      \
		     : "0" (0), "i" (offsetof (struct pthread, member[0])),   \
		     "r" (idx));					      \
     else if (sizeof (__value) == 4)					      \
       asm volatile ("movl %%gs:%P1(,%2,4),%0"				      \
		     : "=r" (__value)					      \
		     : "i" (offsetof (struct pthread, member[0])),	      \
		       "r" (idx));					      \
     else /* 8 */							      \
       {								      \
	 asm volatile  ("movl %%gs:%P1(,%2,8),%%eax\n\t"		      \
			"movl %%gs:4+%P1(,%2,8),%%edx"			      \
			: "=&A" (__value)				      \
			: "i" (offsetof (struct pthread, member[0])),	      \
			  "r" (idx));					      \
       }								      \
     __value; })



/* Set member of the thread descriptor directly.  */
# define THREAD_SETMEM(descr, member, value) \
  ({									      \
     _Static_assert (sizeof (descr->member) == 1			      \
		     || sizeof (descr->member) == 4			      \
		     || sizeof (descr->member) == 8,			      \
		     "size of per-thread data");			      \
     if (sizeof (descr->member) == 1)					      \
       asm volatile ("movb %b0,%%gs:%P1" :				      \
		     : "iq" (value),					      \
		       "i" (offsetof (struct pthread, member)));	      \
     else if (sizeof (descr->member) == 4)				      \
       asm volatile ("movl %0,%%gs:%P1" :				      \
		     : "ir" (value),					      \
		       "i" (offsetof (struct pthread, member)));	      \
     else /* 8 */							      \
       {								      \
	 asm volatile ("movl %%eax,%%gs:%P1\n\t"			      \
		       "movl %%edx,%%gs:%P2" :				      \
		       : "A" ((uint64_t) cast_to_integer (value)),	      \
			 "i" (offsetof (struct pthread, member)),	      \
			 "i" (offsetof (struct pthread, member) + 4));	      \
       }})


/* Same as THREAD_SETMEM, but the member offset can be non-constant.  */
# define THREAD_SETMEM_NC(descr, member, idx, value) \
  ({									      \
     _Static_assert (sizeof (descr->member[0]) == 1			      \
		     || sizeof (descr->member[0]) == 4			      \
		     || sizeof (descr->member[0]) == 8,			      \
		     "size of per-thread data");			      \
     if (sizeof (descr->member[0]) == 1)				      \
       asm volatile ("movb %b0,%%gs:%P1(%2)" :				      \
		     : "iq" (value),					      \
		       "i" (offsetof (struct pthread, member)),		      \
		       "r" (idx));					      \
     else if (sizeof (descr->member[0]) == 4)				      \
       asm volatile ("movl %0,%%gs:%P1(,%2,4)" :			      \
		     : "ir" (value),					      \
		       "i" (offsetof (struct pthread, member)),		      \
		       "r" (idx));					      \
     else /* 8 */							      \
       {								      \
	 asm volatile ("movl %%eax,%%gs:%P1(,%2,8)\n\t"			      \
		       "movl %%edx,%%gs:4+%P1(,%2,8)" :			      \
		       : "A" ((uint64_t) cast_to_integer (value)),	      \
			 "i" (offsetof (struct pthread, member)),	      \
			 "r" (idx));					      \
       }})


/* Set the stack guard field in TCB head.  */
#define THREAD_SET_STACK_GUARD(value) \
  THREAD_SETMEM (THREAD_SELF, header.stack_guard, value)
#define THREAD_COPY_STACK_GUARD(descr) \
  ((descr)->header.stack_guard						      \
   = THREAD_GETMEM (THREAD_SELF, header.stack_guard))


/* Set the pointer guard field in the TCB head.  */
#define THREAD_SET_POINTER_GUARD(value) \
  THREAD_SETMEM (THREAD_SELF, header.pointer_guard, value)
#define THREAD_COPY_POINTER_GUARD(descr) \
  ((descr)->header.pointer_guard					      \
   = THREAD_GETMEM (THREAD_SELF, header.pointer_guard))


/* Get and set the global scope generation counter in the TCB head.  */
#define THREAD_GSCOPE_IN_TCB      1
#define THREAD_GSCOPE_FLAG_UNUSED 0
#define THREAD_GSCOPE_FLAG_USED   1
#define THREAD_GSCOPE_FLAG_WAIT   2
#define THREAD_GSCOPE_RESET_FLAG() \
  do									      \
    { int __res;							      \
      asm volatile ("xchgl %0, %%gs:%P1"				      \
		    : "=r" (__res)					      \
		    : "i" (offsetof (struct pthread, header.gscope_flag)),    \
		      "0" (THREAD_GSCOPE_FLAG_UNUSED));			      \
      if (__res == THREAD_GSCOPE_FLAG_WAIT)				      \
	lll_futex_wake (&THREAD_SELF->header.gscope_flag, 1, LLL_PRIVATE);    \
    }									      \
  while (0)
#define THREAD_GSCOPE_SET_FLAG() \
  THREAD_SETMEM (THREAD_SELF, header.gscope_flag, THREAD_GSCOPE_FLAG_USED)

#endif /* __ASSEMBLER__ */

#endif	/* tls.h */
