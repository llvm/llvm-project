/* Testing s390x PTRACE_SINGLEBLOCK ptrace request.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <elf.h>
#include <support/xunistd.h>
#include <support/check.h>
#include <string.h>
#include <errno.h>

/* Ensure that we use the PTRACE_SINGLEBLOCK definition from glibc ptrace.h
   in tracer_func.  We need the kernel ptrace.h for structs ptrace_area
   and gregset_t.  */
#include <sys/ptrace.h>
static const enum __ptrace_request req_singleblock = PTRACE_SINGLEBLOCK;
#include <asm/ptrace.h>

static void
tracee_func (int pid)
{
  /* Dump the mapping information for manual inspection of the printed
     tracee addresses.  */
  char str[80];
  sprintf (str, "cat /proc/%d/maps", pid);
  puts (str);
  system (str);
  fflush (stdout);

  TEST_VERIFY_EXIT (ptrace (PTRACE_TRACEME) == 0);
  /* Stop tracee.  Afterwards the tracer_func can operate.  */
  kill (pid, SIGSTOP);

  puts ("The PTRACE_SINGLEBLOCK of the tracer will stop after: "
	"brasl %r14,<puts@plt>!");
}

static void
tracer_func (int pid)
{
  unsigned long last_break;
  ptrace_area parea;
  gregset_t regs;
  struct iovec parea2;
  gregset_t regs2;

  int status;
  int ret;
#define MAX_CHARS_IN_BUF 4096
  char buf[MAX_CHARS_IN_BUF + 1];
  size_t buf_count;

  while (1)
    {
      /* Wait for the tracee to be stopped or exited.  */
      wait (&status);
      if (WIFEXITED (status))
	break;

      /* Get information about tracee: gprs, last breaking address.  */
      parea.len = sizeof (regs);
      parea.process_addr = (unsigned long) &regs;
      parea.kernel_addr = 0;
      TEST_VERIFY_EXIT (ptrace (PTRACE_PEEKUSR_AREA, pid, &parea) == 0);
      TEST_VERIFY_EXIT (ptrace (PTRACE_GET_LAST_BREAK, pid, NULL, &last_break)
			== 0);

      parea2.iov_len = sizeof (regs2);
      parea2.iov_base = &regs2;
      TEST_VERIFY_EXIT (ptrace (PTRACE_GETREGSET, pid, NT_PRSTATUS, &parea2)
			== 0);
      TEST_VERIFY_EXIT (parea2.iov_len == sizeof (regs2));

      /* Test if gprs obtained by PTRACE_PEEKUSR_AREA and PTRACE_GETREGESET
	 have the same values.  */
      TEST_VERIFY_EXIT (memcmp (&regs, &regs2, sizeof (regs)) == 0);

      printf ("child IA: %p last_break: %p\n",
	      (void *) regs[1], (void *) last_break);

      /* Execute tracee until next taken branch.

	 Note:
	 Before the commit which introduced this testcase,
	 <glibc>/sysdeps/unix/sysv/linux/s390/sys/ptrace.h
	 uses ptrace-request 12 for PTRACE_GETREGS,
	 but <kernel>/include/uapi/linux/ptrace.h
	 uses 12 for PTRACE_SINGLEBLOCK.

	 The s390 kernel has no support for PTRACE_GETREGS!
	 Thus glibc ptrace.h is adjusted to match kernel ptrace.h.

	 The glibc sys/ptrace.h header contains the identifier
	 PTRACE_SINGLEBLOCK in enum __ptrace_request.  In contrast, the kernel
	 asm/ptrace.h header defines PTRACE_SINGLEBLOCK.

	 This test ensures, that PTRACE_SINGLEBLOCK defined in glibc
	 works as expected.  If the kernel would interpret it as
	 PTRACE_GETREGS, then the tracee will not make any progress
	 and this testcase will time out or the ptrace call will fail with
	 different errors.  */

      /* Ptrace request 12 is first done with data argument pointing to
	 a buffer:
	 -If request 12 is interpreted as PTRACE_GETREGS, it will store the regs
	 to buffer without an error.

	 -If request 12 is interpreted as PTRACE_SINGLEBLOCK, it will fail
	 as data argument is used as signal-number and the address of
	 buf is no valid signal.

	 -If request 12 is not implemented, it will also fail.

	 Here the test expects that the buffer is untouched and an error is
	 returned.  */
      memset (buf, 'a', MAX_CHARS_IN_BUF);
      ret = ptrace (req_singleblock, pid, NULL, buf);
      buf [MAX_CHARS_IN_BUF] = '\0';
      buf_count = strspn (buf, "a");
      TEST_VERIFY_EXIT (buf_count == MAX_CHARS_IN_BUF);
      TEST_VERIFY_EXIT (ret == -1);

      /* If request 12 is interpreted as PTRACE_GETREGS, the first ptrace
	 call will touch the buffer which is detected by this test.  */
      errno = 0;
      ret = ptrace (req_singleblock, pid, NULL, NULL);
      if (ret == 0)
	{
	  /* The kernel has support for PTRACE_SINGLEBLOCK ptrace request. */
	  TEST_VERIFY_EXIT (errno == 0);
	}
      else
	{
	  /* The kernel (< 3.15) has no support for PTRACE_SINGLEBLOCK ptrace
	     request. */
	  TEST_VERIFY_EXIT (errno == EIO);
	  TEST_VERIFY_EXIT (ret == -1);

	  /* Just continue tracee until it exits normally.  */
	  TEST_VERIFY_EXIT (ptrace (PTRACE_CONT, pid, NULL, NULL) == 0);
	}
    }
}

static int
do_test (void)
{
  int pid;
  pid = xfork ();
  if (pid)
    tracer_func (pid);
  else
    tracee_func (getpid ());

  return EXIT_SUCCESS;
}

#include <support/test-driver.c>
