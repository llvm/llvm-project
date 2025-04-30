/* Capture output from a subprocess.
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

#ifndef SUPPORT_CAPTURE_SUBPROCESS_H
#define SUPPORT_CAPTURE_SUBPROCESS_H

#include <support/xmemstream.h>

struct support_capture_subprocess
{
  struct xmemstream out;
  struct xmemstream err;
  int status;
};

/* Invoke CALLBACK (CLOSURE) in a subprocess and capture standard
   output, standard error, and the exit status.  The out.buffer and
   err.buffer members in the result are null-terminated strings which
   can be examined by the caller (out.out and err.out are NULL).  */
struct support_capture_subprocess support_capture_subprocess
  (void (*callback) (void *), void *closure);

/* Issue FILE with ARGV arguments by using posix_spawn and capture standard
   output, standard error, and the exit status.  The out.buffer and err.buffer
   are handle as support_capture_subprocess.  */
struct support_capture_subprocess support_capture_subprogram
  (const char *file, char *const argv[]);

/* Copy the running program into a setgid binary and run it with CHILD_ID
   argument.  If execution is successful, return the exit status of the child
   program, otherwise return a non-zero failure exit code.  */
int support_capture_subprogram_self_sgid
  (char *child_id);

/* Deallocate the subprocess data captured by
   support_capture_subprocess.  */
void support_capture_subprocess_free (struct support_capture_subprocess *);

enum support_capture_allow
{
  /* No output is allowed.  */
  sc_allow_none = 0x01,
  /* Output to stdout is permitted.  */
  sc_allow_stdout = 0x02,
  /* Output to standard error is permitted.  */
  sc_allow_stderr = 0x04,
};

/* Check that the subprocess exited and that only the allowed outputs
   happened.  If STATUS_OR_SIGNAL is nonnegative, it is the expected
   (decoded) exit status of the process, as returned by WEXITSTATUS.
   If STATUS_OR_SIGNAL is negative, -STATUS_OR_SIGNAL is the expected
   termination signal, as returned by WTERMSIG.  ALLOWED is a
   combination of support_capture_allow flags.  Report errors under
   the CONTEXT message.  */
void support_capture_subprocess_check (struct support_capture_subprocess *,
                                       const char *context,
                                       int status_or_signal, int allowed)
  __attribute__ ((nonnull (1, 2)));

#endif /* SUPPORT_CAPTURE_SUBPROCESS_H */
