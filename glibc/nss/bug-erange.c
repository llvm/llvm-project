/* Test case for gethostbyname_r bug when buffer expansion required.  */

#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int
main (void)
{
  const char *host = "www.gnu.org";

  /* This code approximates the example code in the library manual.  */

  struct hostent hostbuf, *hp;
  size_t hstbuflen;
  char *tmphstbuf;
  int res;
  int herr;

  hstbuflen = 16;		/* Make it way small to ensure ERANGE.  */
  /* Allocate buffer, remember to free it to avoid memory leakage.  */
  tmphstbuf = malloc (hstbuflen);

  while ((res = gethostbyname_r (host, &hostbuf, tmphstbuf, hstbuflen,
                                 &hp, &herr)) == ERANGE)
    {
      /* Enlarge the buffer.  */
      hstbuflen *= 2;
      tmphstbuf = realloc (tmphstbuf, hstbuflen);
    }

  if (res != 0 || hp == NULL)
    {
      printf ("gethostbyname_r failed: %s (errno: %m)\n", strerror (res));

      if (access ("/etc/resolv.conf", R_OK))
	{
	  puts ("DNS probably not set up");
	  return 0;
	}

      return 1;
    }

  printf ("Got: %s %s\n", hp->h_name,
	  inet_ntoa (*(struct in_addr *) hp->h_addr));
  return 0;
}
