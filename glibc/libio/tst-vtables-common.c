/* Test for libio vtables and their validation.  Common code.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* This test provides some coverage for how various stdio functions
   use the vtables in FILE * objects.  The focus is mostly on which
   functions call which methods, not so much on validating data
   processing.  An initial series of tests check that custom vtables
   do not work without activation through _IO_init.

   Note: libio vtables are deprecated feature.  Do not use this test
   as a documentation source for writing custom vtables.  See
   fopencookie for a different way of creating custom stdio
   streams.  */

#include <stdbool.h>
#include <string.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

#include "libioP.h"

/* Data shared between the test subprocess and the test driver in the
   parent.  Note that *shared is reset at the start of the check_call
   function.  */
struct shared
{
  /* Expected file pointer for method calls.  */
  FILE *fp;

  /* If true, assume that a call to _IO_init is needed to enable
     custom vtables.  */
  bool initially_disabled;

  /* Requested return value for the methods which have one.  */
  int return_value;

  /* A value (usually a character) recorded by some of the methods
     below.  */
  int value;

  /* Likewise, for some data.  */
  char buffer[16];
  size_t buffer_length;

  /* Total number of method calls.  */
  unsigned int calls;

  /* Individual method call counts.  */
  unsigned int calls_finish;
  unsigned int calls_overflow;
  unsigned int calls_underflow;
  unsigned int calls_uflow;
  unsigned int calls_pbackfail;
  unsigned int calls_xsputn;
  unsigned int calls_xsgetn;
  unsigned int calls_seekoff;
  unsigned int calls_seekpos;
  unsigned int calls_setbuf;
  unsigned int calls_sync;
  unsigned int calls_doallocate;
  unsigned int calls_read;
  unsigned int calls_write;
  unsigned int calls_seek;
  unsigned int calls_close;
  unsigned int calls_stat;
  unsigned int calls_showmanyc;
  unsigned int calls_imbue;
} *shared;

/* Method implementations which increment the counters in *shared.  */

static void
log_method (FILE *fp, const char *name)
{
  if (test_verbose > 0)
    printf ("info: %s (%p) called\n", name, fp);
}

static void
method_finish (FILE *fp, int dummy)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_finish;
}

static int
method_overflow (FILE *fp, int ch)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_overflow;
  shared->value = ch;
  return shared->return_value;
}

static int
method_underflow (FILE *fp)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_underflow;
  return shared->return_value;
}

static int
method_uflow (FILE *fp)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_uflow;
  return shared->return_value;
}

static int
method_pbackfail (FILE *fp, int ch)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_pbackfail;
  shared->value = ch;
  return shared->return_value;
}

static size_t
method_xsputn (FILE *fp, const void *data, size_t n)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_xsputn;

  size_t to_copy = n;
  if (n > sizeof (shared->buffer))
    to_copy = sizeof (shared->buffer);
  memcpy (shared->buffer, data, to_copy);
  shared->buffer_length = to_copy;
  return to_copy;
}

static size_t
method_xsgetn (FILE *fp, void *data, size_t n)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_xsgetn;
  return 0;
}

static off64_t
method_seekoff (FILE *fp, off64_t offset, int dir, int mode)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_seekoff;
  return shared->return_value;
}

static off64_t
method_seekpos (FILE *fp, off64_t offset, int mode)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_seekpos;
  return shared->return_value;
}

static FILE *
method_setbuf (FILE *fp, char *buffer, ssize_t length)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_setbuf;
  return fp;
}

static int
method_sync (FILE *fp)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_sync;
  return shared->return_value;
}

static int
method_doallocate (FILE *fp)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_doallocate;
  return shared->return_value;
}

static ssize_t
method_read (FILE *fp, void *data, ssize_t length)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_read;
  return shared->return_value;
}

static ssize_t
method_write (FILE *fp, const void *data, ssize_t length)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_write;
  return shared->return_value;
}

static off64_t
method_seek (FILE *fp, off64_t offset, int mode)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_seek;
  return shared->return_value;
}

static int
method_close (FILE *fp)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_close;
  return shared->return_value;
}

static int
method_stat (FILE *fp, void *buffer)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_stat;
  return shared->return_value;
}

static int
method_showmanyc (FILE *fp)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_showmanyc;
  return shared->return_value;
}

static void
method_imbue (FILE *fp, void *locale)
{
  log_method (fp, __func__);
  TEST_VERIFY (fp == shared->fp);
  ++shared->calls;
  ++shared->calls_imbue;
}

/* Our custom vtable.  */

static const struct _IO_jump_t jumps =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT (finish, method_finish),
  JUMP_INIT (overflow, method_overflow),
  JUMP_INIT (underflow, method_underflow),
  JUMP_INIT (uflow, method_uflow),
  JUMP_INIT (pbackfail, method_pbackfail),
  JUMP_INIT (xsputn, method_xsputn),
  JUMP_INIT (xsgetn, method_xsgetn),
  JUMP_INIT (seekoff, method_seekoff),
  JUMP_INIT (seekpos, method_seekpos),
  JUMP_INIT (setbuf, method_setbuf),
  JUMP_INIT (sync, method_sync),
  JUMP_INIT (doallocate, method_doallocate),
  JUMP_INIT (read, method_read),
  JUMP_INIT (write, method_write),
  JUMP_INIT (seek, method_seek),
  JUMP_INIT (close, method_close),
  JUMP_INIT (stat, method_stat),
  JUMP_INIT (showmanyc, method_showmanyc),
  JUMP_INIT (imbue, method_imbue)
};

/* Our file implementation.  */

struct my_file
{
  FILE f;
  const struct _IO_jump_t *vtable;
};

struct my_file
my_file_create (void)
{
  return (struct my_file)
    {
      /* Disable locking, so that we do not have to set up a lock
         pointer.  */
      .f._flags =  _IO_USER_LOCK,

      /* Copy the offset from the an initialized handle, instead of
         figuring it out from scratch.  */
      .f._vtable_offset = stdin->_vtable_offset,

      .vtable = &jumps,
    };
}

/* Initial tests which do not enable vtable compatibility.  */

/* Inhibit GCC optimization of fprintf.  */
typedef int (*fprintf_type) (FILE *, const char *, ...);
static const volatile fprintf_type fprintf_ptr = &fprintf;

static void
without_compatibility_fprintf (void *closure)
{
  /* This call should abort.  */
  fprintf_ptr (shared->fp, " ");
  _exit (1);
}

static void
without_compatibility_fputc (void *closure)
{
  /* This call should abort.  */
  fputc (' ', shared->fp);
  _exit (1);
}

static void
without_compatibility_fgetc (void *closure)
{
  /* This call should abort.  */
  fgetc (shared->fp);
  _exit (1);
}

static void
without_compatibility_fflush (void *closure)
{
  /* This call should abort.  */
  fflush (shared->fp);
  _exit (1);
}

static void
check_for_termination (const char *name, void (*callback) (void *))
{
  struct my_file file = my_file_create ();
  shared->fp = &file.f;
  shared->return_value = -1;
  shared->calls = 0;
  struct support_capture_subprocess proc
    = support_capture_subprocess (callback, NULL);
  support_capture_subprocess_check (&proc, name, -SIGABRT,
                                    sc_allow_stderr);
  const char *message
    = "Fatal error: glibc detected an invalid stdio handle\n";
  TEST_COMPARE_BLOB (proc.err.buffer, proc.err.length,
                     message, strlen (message));
  TEST_COMPARE (shared->calls, 0);
  support_capture_subprocess_free (&proc);
}

/* The test with vtable validation disabled.  */

/* This function does not have a prototype in libioP.h to prevent
   accidental use from within the library (which would disable vtable
   verification).  */
void _IO_init (FILE *fp, int flags);

static void
with_compatibility_fprintf (void *closure)
{
  TEST_COMPARE (fprintf_ptr (shared->fp, "A%sCD", "B"), 4);
  TEST_COMPARE (shared->calls, 3);
  TEST_COMPARE (shared->calls_xsputn, 3);
  TEST_COMPARE_BLOB (shared->buffer, shared->buffer_length,
                     "CD", 2);
}

static void
with_compatibility_fputc (void *closure)
{
  shared->return_value = '@';
  TEST_COMPARE (fputc ('@', shared->fp), '@');
  TEST_COMPARE (shared->calls, 1);
  TEST_COMPARE (shared->calls_overflow, 1);
  TEST_COMPARE (shared->value, '@');
}

static void
with_compatibility_fgetc (void *closure)
{
  shared->return_value = 'X';
  TEST_COMPARE (fgetc (shared->fp), 'X');
  TEST_COMPARE (shared->calls, 1);
  TEST_COMPARE (shared->calls_uflow, 1);
}

static void
with_compatibility_fflush (void *closure)
{
  TEST_COMPARE (fflush (shared->fp), 0);
  TEST_COMPARE (shared->calls, 1);
  TEST_COMPARE (shared->calls_sync, 1);
}

/* Call CALLBACK in a subprocess, after setting up a custom file
   object and updating shared->fp.  */
static void
check_call (const char *name, void (*callback) (void *),
            bool initially_disabled)
{
  *shared = (struct shared)
    {
     .initially_disabled = initially_disabled,
    };

  /* Set up a custom file object.  */
  struct my_file file = my_file_create ();
  shared->fp = &file.f;
  if (shared->initially_disabled)
    _IO_init (shared->fp, file.f._flags);

  if (test_verbose > 0)
    printf ("info: calling test %s\n", name);
  support_isolate_in_subprocess (callback, NULL);
}

/* Run the tests.  INITIALLY_DISABLED indicates whether custom vtables
   are disabled when the test starts.  */
static int
run_tests (bool initially_disabled)
{
  /* The test relies on fatal error messages being printed to standard
     error.  */
  setenv ("LIBC_FATAL_STDERR_", "1", 1);

  shared = support_shared_allocate (sizeof (*shared));
  shared->initially_disabled = initially_disabled;

  if (initially_disabled)
    {
      check_for_termination ("fprintf", without_compatibility_fprintf);
      check_for_termination ("fputc", without_compatibility_fputc);
      check_for_termination ("fgetc", without_compatibility_fgetc);
      check_for_termination ("fflush", without_compatibility_fflush);
    }

  check_call ("fprintf", with_compatibility_fprintf, initially_disabled);
  check_call ("fputc", with_compatibility_fputc, initially_disabled);
  check_call ("fgetc", with_compatibility_fgetc, initially_disabled);
  check_call ("fflush", with_compatibility_fflush, initially_disabled);

  support_shared_free (shared);
  shared = NULL;

  return 0;
}
