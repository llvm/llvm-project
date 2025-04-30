#include <netdb.h>
#include <stdio.h>
#include <sys/socket.h>

static int
do_test (void)
{
  int retval = 0;

  struct sockaddr_in6 s;
  s.sin6_family = AF_INET6;
  s.sin6_port = htons (80);
  s.sin6_flowinfo = 0;
  s.sin6_addr = (struct in6_addr) IN6ADDR_ANY_INIT;
  s.sin6_scope_id = 0;
  char buf[1000];
  buf[0] = '\0';
  int r = getnameinfo((struct sockaddr *) &s, sizeof (s), buf, sizeof (buf),
		      NULL, 0, NI_NUMERICSERV);
  printf("r = %d, buf = \"%s\"\n", r, buf);
  if (r != 0)
    {
      puts ("failed without NI_NAMEREQD");
      retval = 1;
    }

  buf[0] = '\0';
  r = getnameinfo((struct sockaddr *) &s, sizeof (s), buf, sizeof (buf),
		  NULL, 0, NI_NUMERICSERV | NI_NAMEREQD);
  printf("r = %d, buf = \"%s\"\n", r, buf);
  if (r != EAI_NONAME)
    {
      puts ("did not fail with EAI_NONAME with NI_NAMEREQD set");
      retval = 1;
    }

  return retval;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
