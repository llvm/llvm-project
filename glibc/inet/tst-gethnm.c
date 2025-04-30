/* Based on a test case by grd@algonet.se.  */

#include <netdb.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static int
do_test (void)
{
  struct hostent *ent;
  struct in_addr hostaddr;
  int result = 0;

  inet_aton ("127.0.0.1", (struct in_addr *) &hostaddr.s_addr);
  ent = gethostbyaddr (&hostaddr, sizeof (hostaddr), AF_INET);
  if (ent == NULL)
    puts ("gethostbyaddr (...) == NULL");
  else
    {
      puts ("Using gethostbyaddr(..):");
      printf ("h_name: %s\n", ent->h_name);

      if (ent->h_aliases == NULL)
	puts ("ent->h_aliases == NULL");
      else
	printf ("h_aliases[0]: %s\n", ent->h_aliases[0]);
    }

  ent = gethostbyname ("127.0.0.1");
  if (ent == NULL)
    {
      puts ("gethostbyname (\"127.0.0.1\") == NULL");
      result = 1;
    }
  else
    {
      printf ("\nNow using gethostbyname(..):\n");
      printf ("h_name: %s\n", ent->h_name);
      if (strcmp (ent->h_name, "127.0.0.1") != 0)
	{
	  puts ("ent->h_name != \"127.0.0.1\"");
	  result = 1;
	}

      if (ent->h_aliases == NULL)
	{
	  puts ("ent->h_aliases == NULL");
	  result = 1;
	}
      else
	{
	  printf ("h_aliases[0]: %s\n", ent->h_aliases[0]);
	  result |= ent->h_aliases[0] != NULL;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
