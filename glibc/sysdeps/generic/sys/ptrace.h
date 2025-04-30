/* `ptrace' debugger support interface.  Generic version; constants are common.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef	_PTRACE_H

#define	_PTRACE_H	1
#include <features.h>

__BEGIN_DECLS


/* Type of the REQUEST argument to `ptrace.'  */
enum __ptrace_request
{
  /* Indicate that the process making this request should be traced.
     All signals received by this process can be intercepted by its
     parent, and its parent can use the other `ptrace' requests.  */
  PTRACE_TRACEME = 0,
#define PT_TRACE_ME PTRACE_TRACEME

  /* Return the word in the process's text space at address ADDR.  */
  PTRACE_PEEKTEXT,
#define PT_READ_I PTRACE_PEEKTEXT

  /* Return the word in the process's data space at address ADDR.  */
  PTRACE_PEEKDATA,
#define PT_READ_D PTRACE_PEEKDATA

  /* Return the word in the process's user area at offset ADDR.  */
  PTRACE_PEEKUSER,
#define PT_READ_U PTRACE_PEEKUSER

  /* Write the word DATA into the process's text space at address ADDR.  */
  PTRACE_POKETEXT,
#define PT_WRITE_I PTRACE_POKETEXT

  /* Write the word DATA into the process's data space at address ADDR.  */
  PTRACE_POKEDATA,
#define PT_WRITE_D PTRACE_POKEDATA

  /* Write the word DATA into the process's user space at offset ADDR.  */
  PTRACE_POKEUSER,
#define PT_WRITE_U PTRACE_POKEUSER

  /* Continue the process.  */
  PTRACE_CONT,
#define PT_CONTINUE PTRACE_CONT

  /* Kill the process.  */
  PTRACE_KILL,
#define PT_KILL PTRACE_KILL

  /* Single step the process.
     This is not supported on all machines.  */
  PTRACE_SINGLESTEP,
#define PT_STEP PTRACE_SINGLESTEP

  /* Attach to a process that is already running. */
  PTRACE_ATTACH,
#define PT_ATTACH PTRACE_ATTACH

  /* Detach from a process attached to with PTRACE_ATTACH.  */
  PTRACE_DETACH,
#define PT_DETACH PTRACE_DETACH

  /* Get the process's registers (not including floating-point registers)
     and put them in the `struct regs' (see <machine/regs.h>) at ADDR.  */
  PTRACE_GETREGS = 12,

  /* Set the process's registers (not including floating-point registers)
     to the contents of the `struct regs' (see <machine/regs.h>) at ADDR.  */
  PTRACE_SETREGS,

  /* Get the process's floating point registers and put them
     in the `struct fp_status' (see <machine/regs.h>) at ADDR.  */
  PTRACE_GETFPREGS = 14,

  /* Set the process's floating point registers to the contents
     of the `struct fp_status' (see <machine/regs.h>) at ADDR.  */
  PTRACE_SETFPREGS,

  /* Read DATA bytes from the process's data space at address ADDR.
     Put the result starting at address ADDR2 in the caller's
     address space.  */
  PTRACE_READDATA = 16,

  /* Write DATA bytes from ADDR2 in the caller's address space into
     the process's data space at address ADDR.  */
  PTRACE_WRITEDATA,

  /* Read DATA bytes from the process's text space at address ADDR.
     Put the result starting at address ADDR2 in the caller's
     address space.  */
  PTRACE_READTEXT = 18,

  /* Write DATA bytes from ADDR2 in the caller's address space into
     the process's text space at address ADDR.  */
  PTRACE_WRITETEXT,

  /* Read the floating-point accelerator unit registers and
     put them into the `struct fpa_regs' (see <machine/regs.h>) at ADDR.  */
  PTRACE_GETFPAREGS = 20,

  /* Write the floating-point accelerator unit registers from
     the contents of the `struct fpa_regs' at ADDR.  */
  PTRACE_SETFPAREGS
};

/* Perform process tracing functions.  REQUEST is one of the values
   above, and determines the action to be taken.
   For all requests except PTRACE_TRACEME, PID specifies the process to be
   traced.

   PID and the other arguments described above for the various requests should
   appear (those that are used for the particular request) as:
     pid_t PID, void *ADDR, int DATA, void *ADDR2
   after REQUEST.  */
extern int ptrace (enum __ptrace_request __request, ...);

__END_DECLS

#endif /* ptrace.h */
