/* Test case by Sam Varshavchik <mrsam@courier-mta.com>.  */
#include <mcheck.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>

static int
do_test (void)
{
  struct addrinfo hints, *res;
  int i, ret;

  mtrace ();
  for (i = 0; i < 100; i++)
    {
      memset (&hints, 0, sizeof (hints));
      hints.ai_family = PF_UNSPEC;
      hints.ai_socktype = SOCK_STREAM;

      ret = getaddrinfo ("www.gnu.org", "http", &hints, &res);

      if (ret)
	FAIL_EXIT1 ("%s\n", gai_strerror (ret));

      freeaddrinfo (res);
    }
  return 0;
}

#include <support/test-driver.c>
