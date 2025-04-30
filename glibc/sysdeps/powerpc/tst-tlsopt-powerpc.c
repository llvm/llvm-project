/* glibc test for __tls_get_addr optimization.  */

static int
do_test (void)
{
  extern int tls_get_addr_opt_test (void);

  return tls_get_addr_opt_test ();
}

#include <support/test-driver.c>
