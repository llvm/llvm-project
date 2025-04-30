#include <stdio.h>
#include <string.h>
#include <netinet/ether.h>


static int
do_test (void)
{
  struct ether_addr a;
  char buf[1000];
  if (ether_line ("00:01:02:03:04:05       aaaaa   \n", &a, buf) != 0)
    {
      puts ("ether_line failed");
      return 1;
    }

  int res = 0;
  int i;
  for (i = 0; i < ETH_ALEN; ++i)
    {
      printf ("%02x%s",
	      (int) a.ether_addr_octet[i], i + 1 == ETH_ALEN ? "" : ":");
      if (a.ether_addr_octet[i] != i)
	{
	  printf ("octet %d is %d, expected %d\n",
		  i, (int) a.ether_addr_octet[i], i);
	  res = 1;
	}
    }

  printf (" \"%s\"\n", buf);
  res |= strcmp (buf, "aaaaa") != 0;

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
