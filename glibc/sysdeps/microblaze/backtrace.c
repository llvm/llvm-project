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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysdep.h>
#include <signal.h>
#include <execinfo.h>

extern int
_identify_sighandler (unsigned long fp, unsigned long pc,
                      unsigned long *pprev_fp, unsigned long *pprev_pc,
                      unsigned long *retaddr);

static inline long
get_frame_size (unsigned long instr)
{
  return abs ((short signed) (instr & 0xFFFF));
}

static unsigned long *
find_frame_creation (unsigned long *pc)
{
  int i;

  /* NOTE: Distance to search is arbitrary.
     250 works well for most things,
     750 picks up things like tcp_recvmsg,
     1000 needed for fat_fill_super.  */
  for (i = 0; i < 1000; i++, pc--)
    {
      unsigned long instr;
      unsigned long frame_size;

      instr = *pc;

      /* Is the instruction of the form
         addik r1, r1, foo ? */
      if ((instr & 0xFFFF0000) != 0x30210000)
        continue;

      frame_size = get_frame_size (instr);

      if ((frame_size < 8) || (frame_size & 3))
        return NULL;

      return pc;
    }
  return NULL;
}

static int
lookup_prev_stack_frame (unsigned long fp, unsigned long pc,
                         unsigned long *pprev_fp, unsigned long *pprev_pc,
                         unsigned long *retaddr)
{
  unsigned long *prologue = NULL;

  int is_signalhandler = _identify_sighandler (fp, pc, pprev_fp,
                                               pprev_pc, retaddr);

  if (!is_signalhandler)
    {
      prologue = find_frame_creation ((unsigned long *) pc);

      if (prologue)
        {
          long frame_size = get_frame_size (*prologue);
          *pprev_fp = fp + frame_size;
          if (*retaddr != 0)
            *pprev_pc = *retaddr;
          else
            *pprev_pc = *(unsigned long *) fp;

          *retaddr = 0;
          if (!*pprev_pc || (*pprev_pc & 3))
            prologue=0;
        }
      else
        {
          *pprev_pc = 0;
          *pprev_fp = fp;
          *retaddr = 0;
        }
    }
    return (!*pprev_pc || (*pprev_pc & 3)) ? -1 : 0;
}

int
__backtrace (void **array, int size)
{
  unsigned long pc, fp;
  unsigned long ppc, pfp;
  /* Return address(r15) is required in the signal handler case, since the
     return address of the function which causes the signal may not be
     recorded in the stack.  */
  unsigned long retaddr;

  int count;
  int rc = 0;

  if (size <= 0)
    return 0;

  __asm__ __volatile__ ("mfs %0, rpc"
                        : "=r"(pc));

  __asm__ __volatile__ ("add %0, r1, r0"
                        : "=r"(fp));

  array[0] = (void *) pc;
  retaddr = 0;
  for (count = 1; count < size; count++)
    {
      rc = lookup_prev_stack_frame (fp, pc, &pfp, &ppc, &retaddr);

      fp = pfp;
      pc = ppc;
      array[count] = (void *) pc;
      if (rc)
        return count;
    }
  return count;
}

weak_alias (__backtrace, backtrace)
libc_hidden_def (__backtrace)
