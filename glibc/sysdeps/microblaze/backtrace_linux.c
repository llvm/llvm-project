/* Copyright (C) 2005-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stddef.h>
#include <asm/sigcontext.h>
#include <linux/signal.h>
#include <asm-generic/ucontext.h>
#include <sys/syscall.h>

int
_identify_sighandler (unsigned long fp, unsigned long pc,
                      unsigned long *pprev_fp, unsigned long *pprev_pc,
                      unsigned long *retaddr)
{
  unsigned long *tramp = 0;
  struct ucontext *uc;

  if (*retaddr == 0)
    {
      /* Kernel inserts the tramp between the signal handler frame and the
         caller frame in signal handling.  */
      tramp = (unsigned long *) pc;
      tramp += 2;
      if ((*tramp == (0x31800000 | __NR_rt_sigreturn))
          && (*(tramp+1) == 0xb9cc0008))
        {
          /* Signal handler function argument are:
             int sig_num, siginfo_t * info, void * ucontext
             therefore ucontext is the 3rd argument.  */
          unsigned long ucptr = ((unsigned long) tramp
                                 - sizeof (struct ucontext));
          uc = (struct ucontext *) ucptr;
          *pprev_pc = uc->uc_mcontext.regs.pc;
          /* Need to record the return address since the return address of the
             function which causes this signal may not be recorded in the
             stack.  */
          *pprev_fp = uc->uc_mcontext.regs.r1;
          *retaddr = uc->uc_mcontext.regs.r15;
          /* It is a signal handler.  */
          return 1;
        }
    }
  return 0;
}
