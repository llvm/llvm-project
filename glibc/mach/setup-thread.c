/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <mach.h>
#include <thread_state.h>
#include <string.h>
#include <mach/machine/vm_param.h>
#include <ldsodefs.h>
#include "sysdep.h"		/* Defines stack direction.  */

#define	STACK_SIZE	(16 * 1024 * 1024) /* 16MB, arbitrary.  */

/* Give THREAD a stack and set it to run at PC when resumed.
   If *STACK_SIZE is nonzero, that size of stack is allocated.
   If *STACK_BASE is nonzero, that stack location is used.
   If STACK_BASE is not null it is filled in with the chosen stack base.
   If STACK_SIZE is not null it is filled in with the chosen stack size.
   Regardless, an extra page of red zone is allocated off the end; this
   is not included in *STACK_SIZE.  */

kern_return_t
__mach_setup_thread (task_t task, thread_t thread, void *pc,
		     vm_address_t *stack_base, vm_size_t *stack_size)
{
  kern_return_t error;
  struct machine_thread_state ts;
  mach_msg_type_number_t tssize = MACHINE_THREAD_STATE_COUNT;
  vm_address_t stack;
  vm_size_t size;
  int anywhere;

  size = stack_size ? *stack_size ? : STACK_SIZE : STACK_SIZE;
  stack = stack_base ? *stack_base ? : 0 : 0;
  anywhere = !stack_base || !*stack_base;

  error = __vm_allocate (task, &stack, size + __vm_page_size, anywhere);
  if (error)
    return error;

  if (stack_size)
    *stack_size = size;

  memset (&ts, 0, sizeof (ts));
  MACHINE_THREAD_STATE_SET_PC (&ts, pc);
#ifdef STACK_GROWTH_DOWN
  if (stack_base)
    *stack_base = stack + __vm_page_size;
  ts.SP = stack + __vm_page_size + size;
#elif defined (STACK_GROWTH_UP)
  if (stack_base)
    *stack_base = stack;
  ts.SP = stack;
  stack += size;
#else
  #error stack direction unknown
#endif

  /* Create the red zone.  */
  if (error = __vm_protect (task, stack, __vm_page_size, 0, VM_PROT_NONE))
    return error;

  return __thread_set_state (thread, MACHINE_NEW_THREAD_STATE_FLAVOR,
			     (natural_t *) &ts, tssize);
}

weak_alias (__mach_setup_thread, mach_setup_thread)

/* Give THREAD a TLS area.  */
kern_return_t
__mach_setup_tls (thread_t thread)
{
  kern_return_t error;
  struct machine_thread_state ts;
  mach_msg_type_number_t tssize = MACHINE_THREAD_STATE_COUNT;
  tcbhead_t *tcb;

  tcb = _dl_allocate_tls (NULL);
  if (tcb == NULL)
    return KERN_RESOURCE_SHORTAGE;

  if (error = __thread_get_state (thread, MACHINE_THREAD_STATE_FLAVOR,
			     (natural_t *) &ts, &tssize))
    return error;
  assert (tssize == MACHINE_THREAD_STATE_COUNT);

  _hurd_tls_new (thread, &ts, tcb);

  error = __thread_set_state (thread, MACHINE_THREAD_STATE_FLAVOR,
			      (natural_t *) &ts, tssize);
  return error;
}

weak_alias (__mach_setup_tls, mach_setup_tls)
