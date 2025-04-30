/* glibc test for static NSS.  */
#include <stdio.h>

#include <support/support.h>

static int
do_test (void)
{
  struct passwd *pw;

  pw = getpwuid(0);
  return pw == NULL;
}


#include <support/test-driver.c>
