/* Definition for thread-local data handling.  nptl/x86_64 version.
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

#ifndef __ASSEMBLER__
# include <asm/prctl.h>	/* For ARCH_SET_FS.  */
# include <stdbool.h>
# include <stddef.h>
# include <stdint.h>
# include <stdlib.h>
# include <sysdep.h>
# include <libc-pointer-arith.h> /* For cast_to_integer.  */
# include <kernel-features.h>
# include <dl-dtv.h>

/* Replacement type for __m128 since this file is included by ld.so,
   which is compiled with -mno-sse.  It must not change the alignment
   of rtld_savespace_sse.  */
typedef struct
{
  int i[4];
} __128bits;


typedef struct
{
  void *tcb;		/* Pointer to the TCB.  Not necessarily the
			   thread descriptor used by libpthread.  */
  dtv_t *dtv;
  void *self;		/* Pointer to the thread descriptor.  */
  int multiple_threads;
  int gscope_flag;
  uintptr_t sysinfo;
  uintptr_t stack_guard;
  uintptr_t pointer_guard;
  unsigned long int unused_vgetcpu_cache[2];
  /* Bit 0: X86_FEATURE_1_IBT.
     Bit 1: X86_FEATURE_1_SHSTK.
   */
  unsigned int feature_1;
  int __glibc_unused1;
  /* Reservation of some values for the TM ABI.  */
  void *__private_tm[4];
  /* GCC split stack support.  */
  void *__private_ss;
  /* The lowest address of shadow stack,  */
  unsigned long long int ssp_base;
  /* Must be kept even if it is no longer used by glibc since programs,
     like AddressSanitizer, depend on the size of tcbhead_t.  */
  __128bits __glibc_unused2[8][4] __attribute__ ((aligned (32)));

  void *__padding[8];
} tcbhead_t;

# ifdef __ILP32__
/* morestack.S in libgcc uses offset 0x40 to access __private_ss,   */
_Static_assert (offsetof (tcbhead_t, __private_ss) == 0x40,
		"offset of __private_ss != 0x40");
/* NB: ssp_base used to be "long int __glibc_reserved2", which was
   changed from 32 bits to 64 bits.  Make sure that the offset of the
   next field, __glibc_unused2, is unchanged.  */
_Static_assert (offsetof (tcbhead_t, __glibc_unused2) == 0x60,
		"offset of __glibc_unused2 != 0x60");
# else
/* morestack.S in libgcc uses offset 0x70 to access __private_ss,   */
_Static_assert (offsetof (tcbhead_t, __private_ss) == 0x70,
		"offset of __private_ss != 0x70");
_Static_assert (offsetof (tcbhead_t, __glibc_unused2) == 0x80,
		"offset of __glibc_unused2 != 0x80");
# endif

#else /* __ASSEMBLER__ */
# include <tcb-offsets.h>
#endif


/* Alignment requirement for the stack.  */
#define STACK_ALIGN	16


#ifndef __ASSEMBLER__
/* Get system call information.  */
# include <sysdep.h>

#define LOCK_PREFIX "lock;"

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


/* Code to initially initialize the thread pointer.  This might need
   special attention since 'errno' is not yet available and if the
   operation can cause a failure 'errno' must not be touched.

   We have to make the syscall for both uses of the macro since the
   address might be (and probably is) different.  */
# define TLS_INIT_TP(thrdescr) \
  ({ void *_thrdescr = (thrdescr);					      \
     tcbhead_t *_head = _thrdescr;					      \
     int _result;							      \
									      \
     _head->tcb = _thrdescr;						      \
     /* For now the thread descriptor is at the same address.  */	      \
     _head->self = _thrdescr;						      \
									      \
     /* It is a simple syscall to set the %fs value for the thread.  */	      \
     asm volatile ("syscall"						      \
		   : "=a" (_result)					      \
		   : "0" ((unsigned long int) __NR_arch_prctl),		      \
		     "D" ((unsigned long int) ARCH_SET_FS),		      \
		     "S" (_thrdescr)					      \
		   : "memory", "cc", "r11", "cx");			      \
									      \
    _result ? "cannot set %fs base address for thread-local storage" : 0;     \
  })

# define TLS_DEFINE_INIT_TP(tp, pd) void *tp = (pd)


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
  (*(struct pthread *__seg_fs *) offsetof (struct pthread, header.self))
# else
#  define THREAD_SELF \
  ({ struct pthread *__self;						      \
     asm ("mov %%fs:%c1,%0" : "=r" (__self)				      \
	  : "i" (offsetof (struct pthread, header.self)));	 	      \
     __self;})
# endif

/* Magic for libthread_db to know how to do THREAD_SELF.  */
# define DB_THREAD_SELF_INCLUDE  <sys/reg.h> /* For the FS constant.  */
# define DB_THREAD_SELF CONST_THREAD_AREA (64, FS)

/* Read member of the thread descriptor directly.  */
# define THREAD_GETMEM(descr, member) \
  ({ __typeof (descr->member) __value;					      \
     _Static_assert (sizeof (__value) == 1				      \
		     || sizeof (__value) == 4				      \
		     || sizeof (__value) == 8,				      \
		     "size of per-thread data");			      \
     if (sizeof (__value) == 1)						      \
       asm volatile ("movb %%fs:%P2,%b0"				      \
		     : "=q" (__value)					      \
		     : "0" (0), "i" (offsetof (struct pthread, member)));     \
     else if (sizeof (__value) == 4)					      \
       asm volatile ("movl %%fs:%P1,%0"					      \
		     : "=r" (__value)					      \
		     : "i" (offsetof (struct pthread, member)));	      \
     else /* 8 */								      \
       {								      \
	 asm volatile ("movq %%fs:%P1,%q0"				      \
		       : "=r" (__value)					      \
		       : "i" (offsetof (struct pthread, member)));	      \
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
       asm volatile ("movb %%fs:%P2(%q3),%b0"				      \
		     : "=q" (__value)					      \
		     : "0" (0), "i" (offsetof (struct pthread, member[0])),   \
		       "r" (idx));					      \
     else if (sizeof (__value) == 4)					      \
       asm volatile ("movl %%fs:%P1(,%q2,4),%0"				      \
		     : "=r" (__value)					      \
		     : "i" (offsetof (struct pthread, member[0])), "r" (idx));\
     else /* 8 */							      \
       {								      \
	 asm volatile ("movq %%fs:%P1(,%q2,8),%q0"			      \
		       : "=r" (__value)					      \
		       : "i" (offsetof (struct pthread, member[0])),	      \
			 "r" (idx));					      \
       }								      \
     __value; })


/* Loading addresses of objects on x86-64 needs to be treated special
   when generating PIC code.  */
#ifdef __pic__
# define IMM_MODE "nr"
#else
# define IMM_MODE "ir"
#endif


/* Set member of the thread descriptor directly.  */
# define THREAD_SETMEM(descr, member, value) \
  ({									      \
     _Static_assert (sizeof (descr->member) == 1			      \
		     || sizeof (descr->member) == 4			      \
		     || sizeof (descr->member) == 8,			      \
		     "size of per-thread data");			      \
     if (sizeof (descr->member) == 1)					      \
       asm volatile ("movb %b0,%%fs:%P1" :				      \
		     : "iq" (value),					      \
		       "i" (offsetof (struct pthread, member)));	      \
     else if (sizeof (descr->member) == 4)				      \
       asm volatile ("movl %0,%%fs:%P1" :				      \
		     : IMM_MODE (value),				      \
		       "i" (offsetof (struct pthread, member)));	      \
     else /* 8 */							      \
       {								      \
	 /* Since movq takes a signed 32-bit immediate or a register source   \
	    operand, use "er" constraint for 32-bit signed integer constant   \
	    or register.  */						      \
	 asm volatile ("movq %q0,%%fs:%P1" :				      \
		       : "er" ((uint64_t) cast_to_integer (value)),	      \
			 "i" (offsetof (struct pthread, member)));	      \
       }})


/* Same as THREAD_SETMEM, but the member offset can be non-constant.  */
# define THREAD_SETMEM_NC(descr, member, idx, value) \
  ({									      \
     _Static_assert (sizeof (descr->member[0]) == 1			      \
		     || sizeof (descr->member[0]) == 4			      \
		     || sizeof (descr->member[0]) == 8,			      \
		     "size of per-thread data");			      \
     if (sizeof (descr->member[0]) == 1)				      \
       asm volatile ("movb %b0,%%fs:%P1(%q2)" :				      \
		     : "iq" (value),					      \
		       "i" (offsetof (struct pthread, member[0])),	      \
		       "r" (idx));					      \
     else if (sizeof (descr->member[0]) == 4)				      \
       asm volatile ("movl %0,%%fs:%P1(,%q2,4)" :			      \
		     : IMM_MODE (value),				      \
		       "i" (offsetof (struct pthread, member[0])),	      \
		       "r" (idx));					      \
     else /* 8 */							      \
       {								      \
	 /* Since movq takes a signed 32-bit immediate or a register source   \
	    operand, use "er" constraint for 32-bit signed integer constant   \
	    or register.  */						      \
	 asm volatile ("movq %q0,%%fs:%P1(,%q2,8)" :			      \
		       : "er" ((uint64_t) cast_to_integer (value)),	      \
			 "i" (offsetof (struct pthread, member[0])),	      \
			 "r" (idx));					      \
       }})


/* Set the stack guard field in TCB head.  */
# define THREAD_SET_STACK_GUARD(value) \
    THREAD_SETMEM (THREAD_SELF, header.stack_guard, value)
# define THREAD_COPY_STACK_GUARD(descr) \
    ((descr)->header.stack_guard					      \
     = THREAD_GETMEM (THREAD_SELF, header.stack_guard))


/* Set the pointer guard field in the TCB head.  */
# define THREAD_SET_POINTER_GUARD(value) \
  THREAD_SETMEM (THREAD_SELF, header.pointer_guard, value)
# define THREAD_COPY_POINTER_GUARD(descr) \
  ((descr)->header.pointer_guard					      \
   = THREAD_GETMEM (THREAD_SELF, header.pointer_guard))


/* Get and set the global scope generation counter in the TCB head.  */
# define THREAD_GSCOPE_IN_TCB      1
# define THREAD_GSCOPE_FLAG_UNUSED 0
# define THREAD_GSCOPE_FLAG_USED   1
# define THREAD_GSCOPE_FLAG_WAIT   2
# define THREAD_GSCOPE_RESET_FLAG() \
  do									      \
    { int __res;							      \
      asm volatile ("xchgl %0, %%fs:%P1"				      \
		    : "=r" (__res)					      \
		    : "i" (offsetof (struct pthread, header.gscope_flag)),    \
		      "0" (THREAD_GSCOPE_FLAG_UNUSED));			      \
      if (__res == THREAD_GSCOPE_FLAG_WAIT)				      \
	lll_futex_wake (&THREAD_SELF->header.gscope_flag, 1, LLL_PRIVATE);    \
    }									      \
  while (0)
# define THREAD_GSCOPE_SET_FLAG() \
  THREAD_SETMEM (THREAD_SELF, header.gscope_flag, THREAD_GSCOPE_FLAG_USED)

#endif /* __ASSEMBLER__ */

#endif	/* tls.h */
