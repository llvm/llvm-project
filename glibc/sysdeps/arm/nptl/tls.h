/* Definition for thread-local data handling.  NPTL/ARM version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _ARM_NPTL_TLS_H
#define _ARM_NPTL_TLS_H	1

#include <dl-sysdep.h>

#ifndef __ASSEMBLER__
# include <stdbool.h>
# include <stddef.h>
# include <stdint.h>
# include <dl-dtv.h>

#else /* __ASSEMBLER__ */
# include <tcb-offsets.h>
#endif /* __ASSEMBLER__ */


#ifndef __ASSEMBLER__

/* The TP points to the start of the thread blocks.  */
# define TLS_DTV_AT_TP	1
# define TLS_TCB_AT_TP	0

/* Get the thread descriptor definition.  */
# include <nptl/descr.h>

typedef struct
{
  dtv_t *dtv;
  void *private;
} tcbhead_t;

/* This is the size of the initial TCB.  */
# define TLS_INIT_TCB_SIZE	sizeof (tcbhead_t)

/* Alignment requirements for the initial TCB.  */
# define TLS_INIT_TCB_ALIGN	16

/* This is the size of the TCB.  */
# define TLS_TCB_SIZE		sizeof (tcbhead_t)

/* This is the size we need before TCB.  */
# define TLS_PRE_TCB_SIZE	sizeof (struct pthread)

/* Alignment requirements for the TCB.  */
# define TLS_TCB_ALIGN		16

/* Install the dtv pointer.  The pointer passed is to the element with
   index -1 which contain the length.  */
# define INSTALL_DTV(tcbp, dtvp) \
  (((tcbhead_t *) (tcbp))->dtv = (dtvp) + 1)

/* Install new dtv for current thread.  */
# define INSTALL_NEW_DTV(dtv) \
  (THREAD_DTV() = (dtv))

/* Return dtv of given thread descriptor.  */
# define GET_DTV(tcbp) \
  (((tcbhead_t *) (tcbp))->dtv)

# define TLS_DEFINE_INIT_TP(tp, pd) void *tp = (pd) + 1

/* Return the address of the dtv for the current thread.  */
# define THREAD_DTV() \
  (((tcbhead_t *) __builtin_thread_pointer ())->dtv)

/* Return the thread descriptor for the current thread.  */
# define THREAD_SELF \
 ((struct pthread *)__builtin_thread_pointer () - 1)

/* Magic for libthread_db to know how to do THREAD_SELF.  */
# define DB_THREAD_SELF \
  CONST_THREAD_AREA (32, sizeof (struct pthread))

/* Access to data in the thread descriptor is easy.  */
#define THREAD_GETMEM(descr, member) \
  descr->member
#define THREAD_GETMEM_NC(descr, member, idx) \
  descr->member[idx]
#define THREAD_SETMEM(descr, member, value) \
  descr->member = (value)
#define THREAD_SETMEM_NC(descr, member, idx, value) \
  descr->member[idx] = (value)

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
