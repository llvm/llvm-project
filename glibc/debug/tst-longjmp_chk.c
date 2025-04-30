/* Basic test to make sure doing a longjmp to a jmpbuf with an invalid sp
   is caught by the fortification code.  */
#include <errno.h>
#include <fcntl.h>
#include <paths.h>
#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int do_test(void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"


static jmp_buf b;


static void
__attribute__ ((noinline))
f (void)
{
  char buf[1000];
  asm volatile ("" : "=m" (buf));

  if (setjmp (b) != 0)
    {
      puts ("second longjmp succeeded");
      exit (1);
    }
}


static bool expected_to_fail;


static void
handler (int sig)
{
  if (expected_to_fail)
    _exit (0);
  else
    {
      static const char msg[] = "unexpected longjmp failure\n";
      TEMP_FAILURE_RETRY (write (STDOUT_FILENO, msg, sizeof (msg) - 1));
      _exit (1);
    }
}


static int
do_test (void)
{
  set_fortify_handler (handler);


  expected_to_fail = false;

  if (setjmp (b) == 0)
    {
      longjmp (b, 1);
      /* NOTREACHED */
      printf ("first longjmp returned\n");
      return 1;
    }


  expected_to_fail = true;

  f ();
  longjmp (b, 1);

  puts ("second longjmp returned");
  return 1;
}
