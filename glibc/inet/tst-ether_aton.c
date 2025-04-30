#include <stdio.h>
#include <string.h>
#include <netinet/ether.h>

static int
do_test (void)
{
  struct ether_addr *valp, val;
  int result, r;
  char hostname[32], buf[64], *p;

  valp = ether_aton ("12:34:56:78:9a:bc");

  printf ("ether_aton (\"12:34:56:78:9a:bc\") = %hhx:%hhx:%hhx:%hhx:%hhx:%hhx\n",
	  valp->ether_addr_octet[0],
	  valp->ether_addr_octet[1],
	  valp->ether_addr_octet[2],
	  valp->ether_addr_octet[3],
	  valp->ether_addr_octet[4],
	  valp->ether_addr_octet[5]);

  result = (valp->ether_addr_octet[0] != 0x12
	    || valp->ether_addr_octet[1] != 0x34
	    || valp->ether_addr_octet[2] != 0x56
	    || valp->ether_addr_octet[3] != 0x78
	    || valp->ether_addr_octet[4] != 0x9a
	    || valp->ether_addr_octet[5] != 0xbc);

  if ((r = ether_line ("0:c0:f0:46:5f:97 host.ether.com \t# comment",
		       &val, hostname)) == 0)
    {
      ether_ntoa_r (&val, buf);
      p = strchr (buf, '\0');
      *p++ = ' ';
      strcpy (p, hostname);

      printf ("ether_line (\"0:c0:f0:46:5f:97 host.ether.com\") = \"%s\"\n",
	      buf);

      result |= strcmp ("0:c0:f0:46:5f:97 host.ether.com", buf) != 0;
    }
  else
    {
      printf ("ether_line (\"0:c0:f0:46:5f:97 host.ether.com\") = %d\n", r);
      result |= 1;
    }

  r = ether_line ("0:c0:2:d0 foo.bar   ", &val, hostname);
  printf ("ether_line (\"0:c0:2:d0 foo.bar   \") = %d\n", r);
  result |= r != -1;

  r = ether_line ("0:c0:2:d0:1a:2a  ", &val, hostname);
  printf ("ether_line (\"0:c0:2:d0:1a:2a  \") = %d\n", r);
  result |= r != -1;

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
