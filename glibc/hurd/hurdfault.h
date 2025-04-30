/* Declarations for handling faults in the signal thread.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _HURD_FAULT_H
#define _HURD_FAULT_H

#include <hurd/sigpreempt.h>
#include <setjmp.h>

/* Call this before code that might fault in the signal thread; SIGNO is
   the signal expected to possibly arrive.  This behaves like setjmp: it
   returns zero the first time, and returns again nonzero if the signal
   does arrive.  */

#define _hurdsig_catch_fault(sigset, firstcode, lastcode)	\
  (_hurdsig_fault_preemptor.signals = (sigset),			\
   _hurdsig_fault_preemptor.first = (long int) (firstcode),	\
   _hurdsig_fault_preemptor.last = (long int) (lastcode),	\
   setjmp (_hurdsig_fault_env))

/* Call this at the end of a section protected by _hurdsig_catch_fault.  */

#define _hurdsig_end_catch_fault() \
  (_hurdsig_fault_preemptor.signals = 0)

extern jmp_buf _hurdsig_fault_env;
extern struct hurd_signal_preemptor _hurdsig_fault_preemptor;


#define _hurdsig_catch_memory_fault(object) \
  _hurdsig_catch_fault (__sigmask (SIGSEGV) | __sigmask (SIGBUS), \
			(object), (object) + 1)


#endif	/* hurdfault.h */
