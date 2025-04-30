/* POSIX.1 `sigaction' call for Linux/i386.
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

#include <signal.h>
#include <ldsodefs.h>

#define SA_RESTORER 0x04000000

extern void restore_rt (void) asm ("__restore_rt") attribute_hidden;
extern void restore (void) asm ("__restore") attribute_hidden;

#define SET_SA_RESTORER(kact, act)				\
  ({								\
     if (GLRO(dl_sysinfo_dso) == NULL)				\
       {							\
	 (kact)->sa_flags |= SA_RESTORER;			\
         (kact)->sa_restorer = (((act)->sa_flags & SA_SIGINFO)	\
			       ? &restore_rt : &restore);	\
       }							\
     else							\
       (kact)->sa_restorer = NULL;				\
  })

#define RESET_SA_RESTORER(act, kact) \
  (act)->sa_restorer = (kact)->sa_restorer

#include <sysdeps/unix/sysv/linux/libc_sigaction.c>

/* NOTE: Please think twice before making any changes to the bits of
   code below.  GDB needs some intimate knowledge about it to
   recognize them as signal trampolines, and make backtraces through
   signal handlers work right.  Important are both the names
   (__restore and __restore_rt) and the exact instruction sequence.
   If you ever feel the need to make any changes, please notify the
   appropriate GDB maintainer.  */

#define RESTORE(name, syscall) RESTORE2 (name, syscall)
#define RESTORE2(name, syscall) \
asm						\
  (						\
   ".text\n"					\
   "	.align 16\n"				\
   "__" #name ":\n"				\
   "	movl $" #syscall ", %eax\n"		\
   "	int  $0x80"				\
   );

/* The return code for realtime-signals.  */
RESTORE (restore_rt, __NR_rt_sigreturn)

/* For the boring old signals.  */
#undef RESTORE2
#define RESTORE2(name, syscall) \
asm						\
  (						\
   ".text\n"					\
   "	.align 8\n"				\
   "__" #name ":\n"				\
   "	popl %eax\n"				\
   "	movl $" #syscall ", %eax\n"		\
   "	int  $0x80"				\
   );

RESTORE (restore, __NR_sigreturn)
