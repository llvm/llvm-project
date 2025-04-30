#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>


static int
test (unsigned int inaddr, const char *expected)
{
  struct in_addr addr;
  char *res;
  int fail;

  addr.s_addr = htonl (inaddr);
  res = inet_ntoa (addr);
  fail = strcmp (res, expected);

  printf ("%#010x -> \"%s\" -> %s%s\n", inaddr, res,
	  fail ? "fail, expected" : "ok", fail ? expected : "");

  return fail;
}


static int
do_test (void)
{
  int result = 0;

  result |= test (INADDR_LOOPBACK, "127.0.0.1");
  result |= test (INADDR_BROADCAST, "255.255.255.255");
  result |= test (INADDR_ANY, "0.0.0.0");
  result |= test (0xc0060746, "192.6.7.70");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
