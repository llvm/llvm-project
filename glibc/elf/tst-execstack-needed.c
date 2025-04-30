/* Test program for making nonexecutable stacks executable
   on DT_NEEDED load of a DSO that requires executable stacks.  */

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <error.h>

extern void tryme (void);	/* from tst-execstack-mod.so */

static void deeper (void (*f) (void));

static int
do_test (void)
{
  tryme ();

  /* Test that growing the stack region gets new executable pages too.  */
  deeper (&tryme);

  return 0;
}

static void
deeper (void (*f) (void))
{
  char stack[1100 * 1024];
  memfrob (stack, sizeof stack);
  (*f) ();
  memfrob (stack, sizeof stack);
}

#include <support/test-driver.c>
