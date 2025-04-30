#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <rpc/clnt.h>


static int
do_test (void)
{
  struct sockaddr_in ad;
  struct sockaddr_in ad2;
  memset (&ad, '\0', sizeof (ad));
  memset (&ad2, '\0', sizeof (ad2));

  get_myaddress (&ad);

  printf ("addr = %s:%d\n", inet_ntoa (ad.sin_addr), ad.sin_port);

  return memcmp (&ad, &ad2, sizeof (ad)) == 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
