/* Mach thread state definitions for machine-independent code.  Stub version.
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

/* Everything else is called `thread_state', but CMU's header file is
   called `thread_status'.  Oh boy.  */
#include <mach/thread_state.h>

/* Replace <machine> with "i386" or "mips" or whatever.  */

/* This lets the kernel define architecture-specific registers for a new
   thread.  */
#define MACHINE_NEW_THREAD_STATE_FLAVOR	<machine>_NEW_THREAD_STATE
/* This makes the kernel load all architectures-specific registers for the
   thread.  */
#define MACHINE_THREAD_STATE_FLAVOR	<machine>_THREAD_STATE
#define MACHINE_THREAD_STATE_COUNT	<machine>_THREAD_STATE_COUNT

#define machine_thread_state <machine>_thread_state

/* Define these to the member names in `struct <machine>_thread_state'
   for the PC and stack pointer.  */
#define PC ?
#define SP ?

/* This structure should contain all of the different flavors of thread
   state structures which are meaningful for this machine.  Every machine's
   definition of this structure should have a member `int set' which is a
   bit mask (1 << FLAVOR) of the flavors of thread state in the structure
   which are filled in; and a member `struct machine_thread_state basic'.
   On some machines those are the only members (e.g. i386); on others,
   there are several relevant flavors of thread state (e.g. mips).  */
struct machine_thread_all_state
  {
    int set;			/* Mask of bits (1 << FLAVOR).  */
    struct <machine>_thread_state basic;
  };
