/* Invoking main from __libc_start_main.  nptl version.
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

#include <atomic.h>
#include <pthreadP.h>

_Noreturn static void
__libc_start_call_main (int (*main) (int, char **, char ** MAIN_AUXVEC_DECL),
                        int argc, char **argv
#ifdef LIBC_START_MAIN_AUXVEC_ARG
                            , ElfW(auxv_t) *auxvec
#endif
                        )
{
  int result;

  /* Memory for the cancellation buffer.  */
  struct pthread_unwind_buf unwind_buf;

  int not_first_call;
  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* This call results in a -Wstringop-overflow warning because struct
     pthread_unwind_buf is smaller than jmp_buf.  setjmp and longjmp
     do not use anything beyond the common prefix (they never access
     the saved signal mask), so that is a false positive.  */
  DIAG_IGNORE_NEEDS_COMMENT (11, "-Wstringop-overflow=");
#endif
  not_first_call = setjmp ((struct __jmp_buf_tag *) unwind_buf.cancel_jmp_buf);
  DIAG_POP_NEEDS_COMMENT;
  if (__glibc_likely (! not_first_call))
    {
      struct pthread *self = THREAD_SELF;

      /* Store old info.  */
      unwind_buf.priv.data.prev = THREAD_GETMEM (self, cleanup_jmp_buf);
      unwind_buf.priv.data.cleanup = THREAD_GETMEM (self, cleanup);

      /* Store the new cleanup handler info.  */
      THREAD_SETMEM (self, cleanup_jmp_buf, &unwind_buf);

      /* Run the program.  */
      result = main (argc, argv, __environ MAIN_AUXVEC_PARAM);
    }
  else
    {
      /* Remove the thread-local data.  */
      __nptl_deallocate_tsd ();

      /* One less thread.  Decrement the counter.  If it is zero we
         terminate the entire process.  */
      result = 0;
      if (! atomic_decrement_and_test (&__nptl_nthreads))
        /* Not much left to do but to exit the thread, not the process.  */
	while (1)
	  INTERNAL_SYSCALL_CALL (exit, 0);
    }

  exit (result);
}
