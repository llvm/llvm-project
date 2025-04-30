/* This file contains an example test case which shows minimal use of
   the test framework.  Additional testing hooks are described in
   <support/test-driver.c>.  */

/* This function will be called from the test driver.  */
static int
do_test (void)
{
  if (3 == 5)
    /* Indicate failure.  */
    return 1;
  else
    /* Indicate success.  */
    return 0;
}

/* This file references do_test above and contains the definition of
   the main function.  */
#include <support/test-driver.c>
