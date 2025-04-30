/* Definitions for thread-local data handling.  Hurd version.
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
#define _TLS_H

#ifndef __ASSEMBLER__

# include <stddef.h>
# include <stdint.h>
# include <stdbool.h>
# include <sysdep.h>
# include <mach/mig_errors.h>
# include <mach.h>
# include <atomic.h>


/* This is the size of the initial TCB.  */
# define TLS_INIT_TCB_SIZE sizeof (tcbhead_t)

/* Alignment requirements for the initial TCB.  */
# define TLS_INIT_TCB_ALIGN __alignof__ (tcbhead_t)

/* This is the size of the TCB.  */
# define TLS_TCB_SIZE TLS_INIT_TCB_SIZE	/* XXX */

/* Alignment requirements for the TCB.  */
# define TLS_TCB_ALIGN TLS_INIT_TCB_ALIGN /* XXX */


/* Install the dtv pointer.  The pointer passed is to the element with
   index -1 which contain the length.  */
# define INSTALL_DTV(descr, dtvp) \
  ((tcbhead_t *) (descr))->dtv = (dtvp) + 1

/* Return dtv of given thread descriptor.  */
# define GET_DTV(descr) \
  (((tcbhead_t *) (descr))->dtv)

/* Global scope switch support.  */
#define THREAD_GSCOPE_IN_TCB      0
#define THREAD_GSCOPE_GLOBAL
#define THREAD_GSCOPE_SET_FLAG() \
  atomic_exchange_and_add_acq (&GL(dl_thread_gscope_count), 1)
#define THREAD_GSCOPE_RESET_FLAG() \
  do 									      \
    if (atomic_exchange_and_add_rel (&GL(dl_thread_gscope_count), -1) == 1)   \
      lll_wake (GL(dl_thread_gscope_count), 0);				      \
  while (0)
#define THREAD_GSCOPE_WAIT() \
  do 									      \
    {									      \
      int count;							      \
      atomic_write_barrier ();						      \
      while ((count = GL(dl_thread_gscope_count)))			      \
        lll_wait (GL(dl_thread_gscope_count), count, 0);		      \
    }									      \
  while (0)

#endif /* !ASSEMBLER */


#endif /* tls.h */
