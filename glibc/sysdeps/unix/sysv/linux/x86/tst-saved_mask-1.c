/* Test that sigprocmask does not read from the unused part of jmpbuf.
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

#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <setjmpP.h>
#include <support/next_to_fault.h>

#define SIZEOF_SIGSET_T sizeof (__jmp_buf_sigset_t)

static int
do_test (void)
{
  sigjmp_buf sj;
  struct support_next_to_fault sigset_t_buf
    = support_next_to_fault_allocate (SIZEOF_SIGSET_T);
  sigset_t *m_p = (sigset_t *) sigset_t_buf.buffer;
  sigset_t m;

  sigemptyset (&m);
  memcpy (m_p, &m, SIZEOF_SIGSET_T);
  sigprocmask (SIG_SETMASK, m_p, NULL);
  memcpy (&m, m_p, SIZEOF_SIGSET_T);
  if (sigsetjmp (sj, 0) == 0)
    {
      sigaddset (&m, SIGUSR1);
      memcpy (m_p, &m, SIZEOF_SIGSET_T);
      sigprocmask (SIG_SETMASK, m_p, NULL);
      memcpy (&m, m_p, SIZEOF_SIGSET_T);
      siglongjmp (sj, 1);
      return EXIT_FAILURE;
    }
  sigprocmask (SIG_SETMASK, NULL, m_p);
  memcpy (&m, m_p, SIZEOF_SIGSET_T);
  return sigismember (&m, SIGUSR1) ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include <support/test-driver.c>
