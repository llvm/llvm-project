/* Generic definitions for dealing with Mach thread states.
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
#include <mach/thread_status.h>

/* The machine-dependent thread_state.h file can either define these
   macros, or just define PC and SP to the register names.  */

#ifndef MACHINE_THREAD_STATE_SET_PC
#define MACHINE_THREAD_STATE_SET_PC(ts, pc) \
  ((ts)->PC = (unsigned long int) (pc))
#endif
#ifndef MACHINE_THREAD_STATE_SET_SP
#ifdef STACK_GROWTH_UP
#define MACHINE_THREAD_STATE_SET_SP(ts, stack, size) \
  ((ts)->SP = (unsigned long int) (stack))
#else
#define MACHINE_THREAD_STATE_SET_SP(ts, stack, size) \
  ((ts)->SP = (unsigned long int) (stack) + (size))
#endif
#endif

/* This copies architecture-specific bits from the current thread to the new
   thread state.  */
#ifndef MACHINE_THREAD_STATE_FIX_NEW
# define MACHINE_THREAD_STATE_FIX_NEW(ts)
#endif

/* These functions are of use in machine-dependent signal trampoline
   implementations.  */

#include <string.h>		/* size_t, memcpy */
#include <mach/mach_interface.h> /* __thread_get_state */

static inline int
machine_get_state (thread_t thread, struct machine_thread_all_state *state,
		   int flavor, void *stateptr, void *scpptr, size_t size)
{
  if (state->set & (1 << flavor))
    {
      /* Copy the saved state.  */
      memcpy (scpptr, stateptr, size);
      return 1;
    }
  else
    {
      /* No one asked about this flavor of state before; fetch the state
	 directly from the kernel into the sigcontext.  */
      mach_msg_type_number_t got = (size / sizeof (int));
      return (! __thread_get_state (thread, flavor, scpptr, &got)
	      && got == (size / sizeof (int)));
    }
}

static inline int
machine_get_basic_state (thread_t thread,
			 struct machine_thread_all_state *state)
{
  mach_msg_type_number_t count;

  if (state->set & (1 << MACHINE_THREAD_STATE_FLAVOR))
    return 1;

  count = MACHINE_THREAD_STATE_COUNT;
  if (__thread_get_state (thread, MACHINE_THREAD_STATE_FLAVOR,
			  (natural_t *) &state->basic,
			  &count) != KERN_SUCCESS
      || count != MACHINE_THREAD_STATE_COUNT)
    /* What kind of thread?? */
    return 0;			/* XXX */

  state->set |= 1 << MACHINE_THREAD_STATE_FLAVOR;
  return 1;
}
