/* Standard header for all Mach programs.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef	_MACH_H

#define	_MACH_H	1

#include <features.h>

/* Get the basic types used by Mach.  */
#include <mach/mach_types.h>

/* This declares the basic variables and macros everything needs.  */
#include <mach_init.h>

/* This declares all the real system call functions.  */
#include <mach/mach_traps.h>

/* These are MiG-generated headers for the kernel interfaces commonly used.  */
#include <mach/mach_interface.h> /* From <mach/mach.defs>.  */
#include <mach/mach_port.h>
#include <mach/mach_host.h>

/* For the kernel RPCs which have system call shortcut versions,
   the MiG-generated header in fact declares `CALL_rpc' rather than `CALL'.
   This file declares the simple `CALL' functions.  */
#include <mach-shortcuts.h>


/* Receive RPC request messages on RCV_NAME and pass them to DEMUX, which
   decodes them and produces reply messages.  MAX_SIZE is the maximum size
   (in bytes) of the request and reply buffers.  */
extern mach_msg_return_t
__mach_msg_server (boolean_t (*__demux) (mach_msg_header_t *__request,
					 mach_msg_header_t *__reply),
		   mach_msg_size_t __max_size,
		   mach_port_t __rcv_name),
mach_msg_server (boolean_t (*__demux) (mach_msg_header_t *__request,
				       mach_msg_header_t *__reply),
		 mach_msg_size_t __max_size,
		 mach_port_t __rcv_name);

/* Just like `mach_msg_server', but the OPTION and TIMEOUT parameters are
   passed on to `mach_msg'.  */
extern mach_msg_return_t
__mach_msg_server_timeout (boolean_t (*__demux) (mach_msg_header_t *__request,
						 mach_msg_header_t *__reply),
			   mach_msg_size_t __max_size,
			   mach_port_t __rcv_name,
			   mach_msg_option_t __option,
			   mach_msg_timeout_t __timeout),
mach_msg_server_timeout (boolean_t (*__demux) (mach_msg_header_t *__request,
					       mach_msg_header_t *__reply),
			 mach_msg_size_t __max_size,
			 mach_port_t __rcv_name,
			 mach_msg_option_t __option,
			 mach_msg_timeout_t __timeout);


/* Deallocate all port rights and out-of-line memory in MSG. */
extern void
__mach_msg_destroy (mach_msg_header_t *msg),
mach_msg_destroy (mach_msg_header_t *msg);

#include <bits/types/FILE.h>

/* Open a stream on a Mach device.  */
extern FILE *mach_open_devstream (mach_port_t device_port, const char *mode);

/* Give THREAD a stack and set it to run at PC when resumed.
   If *STACK_SIZE is nonzero, that size of stack is allocated.
   If *STACK_BASE is nonzero, that stack location is used.
   If STACK_BASE is not null it is filled in with the chosen stack base.
   If STACK_SIZE is not null it is filled in with the chosen stack size.
   Regardless, an extra page of red zone is allocated off the end; this
   is not included in *STACK_SIZE.  */
kern_return_t __mach_setup_thread (task_t task, thread_t thread, void *pc,
				   vm_address_t *stack_base,
				   vm_size_t *stack_size);
kern_return_t mach_setup_thread (task_t task, thread_t thread, void *pc,
				 vm_address_t *stack_base,
				 vm_size_t *stack_size);

/* Give THREAD a TLS area.  */
kern_return_t __mach_setup_tls (thread_t thread);
kern_return_t mach_setup_tls (thread_t thread);

#endif	/* mach.h */
