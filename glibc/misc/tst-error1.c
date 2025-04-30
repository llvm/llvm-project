#include <error.h>
#include <mcheck.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <libc-diag.h>

static int
do_test (int argc, char *argv[])
{
  mtrace ();
  (void) freopen (argc == 1 ? "/dev/stdout" : argv[1], "a", stderr);
  /* Orient the stream.  */
  fwprintf (stderr, L"hello world\n");
  char buf[20000];
  static const char str[] = "hello world! ";
  for (int i = 0; i < 1000; ++i)
    memcpy (&buf[i * (sizeof (str) - 1)], str, sizeof (str));
  error (0, 0, str);

  /* We're testing a large format string here and need to generate it
     to avoid this source file being ridiculous.  So disable the warning
     about a generated format string.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat-security");

  error (0, 0, buf);
  error (0, 0, buf);

  DIAG_POP_NEEDS_COMMENT;

  error (0, 0, str);
  return 0;
}

#define TEST_FUNCTION do_test (argc, argv)
#include "../test-skeleton.c"
