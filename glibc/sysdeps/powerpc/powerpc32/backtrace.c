/* Return backtrace of current program state.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <execinfo.h>
#include <stddef.h>
#include <string.h>
#include <signal.h>
#include <libc-vdso.h>

/* This is the stack layout we see with every stack frame.
   Note that every routine is required by the ABI to lay out the stack
   like this.

            +----------------+        +-----------------+
    %r1  -> | %r1 last frame--------> | %r1 last frame--->...  --> NULL
            |                |        |                 |
            | (unused)       |        | return address  |
            +----------------+        +-----------------+
*/
struct layout
{
  struct layout *next;
  void *return_address;
};

#define SIGNAL_FRAMESIZE 64

/* Since the signal handler is just like any other function it needs to
   save/restore its LR and it will save it into callers stack frame.
   Since a signal handler doesn't have a caller, the kernel creates a
   dummy frame to make it look like it has a caller.  */
struct signal_frame_32 {
  char               dummy[SIGNAL_FRAMESIZE];
  struct sigcontext  sctx;
  mcontext_t         mctx;
  /* We don't care about the rest, since IP value is at 'mctx' field.  */
};

static inline bool
is_sigtramp_address (void *nip)
{
#ifdef HAVE_SIGTRAMP_RT32
  if (nip == GLRO (dl_vdso_sigtramp_32))
    return true;
#endif
  return false;
}

struct rt_signal_frame_32 {
  char               dummy[SIGNAL_FRAMESIZE + 16];
  siginfo_t          info;
  ucontext_t         uc;
  /* We don't care about the rest, since IP value is at 'uc' field.  */
};

static inline bool
is_sigtramp_address_rt (void * nip)
{
#ifdef HAVE_SIGTRAMP_32
  if (nip == GLRO (dl_vdso_sigtramp_rt32))
    return true;
#endif
  return false;
}

int
__backtrace (void **array, int size)
{
  struct layout *current;
  int count;

  /* Force gcc to spill LR.  */
  asm volatile ("" : "=l"(current));

  /* Get the address on top-of-stack.  */
  asm volatile ("lwz %0,0(1)" : "=r"(current));

  for (				count = 0;
       current != NULL && 	count < size;
       current = current->next, count++)
    {
      gregset_t *gregset = NULL;

      array[count] = current->return_address;

      /* Check if the symbol is the signal trampoline and get the interrupted
       * symbol address from the trampoline saved area.  */
      if (is_sigtramp_address (current->return_address))
	{
	  struct signal_frame_32 *sigframe =
	    (struct signal_frame_32*) current;
          gregset = &sigframe->mctx.gregs;
        }
      else if (is_sigtramp_address_rt (current->return_address))
	{
	  struct rt_signal_frame_32 *sigframe =
            (struct rt_signal_frame_32*) current;
          gregset = &sigframe->uc.uc_mcontext.uc_regs->gregs;
        }
      if (gregset)
	{
	  if (count + 1 == size)
	    break;
	  array[++count] = (void*)((*gregset)[PT_NIP]);
	  current = (void*)((*gregset)[PT_R1]);
	}
    }

  /* It's possible the second-last stack frame can't return
     (that is, it's __libc_start_main), in which case
     the CRT startup code will have set its LR to 'NULL'.  */
  if (count > 0 && array[count-1] == NULL)
    count--;

  return count;
}
weak_alias (__backtrace, backtrace)
libc_hidden_def (__backtrace)
