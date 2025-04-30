#include <netdb.h>
#include <stdio.h>
#include <sys/socket.h>

static int
do_test (void)
{
  int retval = 0;

  struct sockaddr_in s;
  s.sin_family = AF_INET;
  s.sin_port = 80;
  s.sin_addr.s_addr = INADDR_LOOPBACK;
  int r = getnameinfo((struct sockaddr *) &s, sizeof (s), NULL, 0, NULL, 0,
		      NI_NUMERICHOST | NI_NUMERICSERV);
  printf("r = %d\n", r);
  if (r != 0)
    {
      puts ("failed without NI_NAMEREQD");
      retval = 1;
    }

  r = getnameinfo((struct sockaddr *) &s, sizeof (s), NULL, 0, NULL, 0,
		  NI_NUMERICHOST | NI_NUMERICSERV | NI_NAMEREQD);
  printf("r = %d\n", r);
  if (r != EAI_NONAME)
    {
      puts ("did not fail with EAI_NONAME with NI_NAMEREQD set");
      retval = 1;
    }

  return retval;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
