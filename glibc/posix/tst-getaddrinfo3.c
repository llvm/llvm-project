#include <mcheck.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <arpa/nameser.h>


static int
do_test (void)
{
  mtrace ();

  int result = 0;
  struct addrinfo hints;
  struct addrinfo *ai_res;
  int s;

#define T(no, fail, addr, fam, coraddr)					      \
  s = getaddrinfo (addr, NULL, &hints, &ai_res);			      \
  if (s != 0)								      \
    {									      \
      if (s != fail)							      \
	{								      \
	  printf ("getaddrinfo test %d failed: %s\n", no, gai_strerror (s));  \
	  result = 1;							      \
	}								      \
      ai_res = NULL;							      \
    }									      \
  else if (fail)							      \
    {									      \
      printf ("getaddrinfo test %d should have failed but did not\n", no);    \
      result = 1;							      \
    }									      \
  else if (ai_res->ai_family != fam)					      \
    {									      \
      printf ("\
getaddrinfo test %d return address of family %d, expected %d\n",	      \
	      no, ai_res->ai_family, fam);				      \
      result = 1;							      \
    }									      \
  else if (fam == AF_INET)						      \
    {									      \
      if (ai_res->ai_addrlen != sizeof (struct sockaddr_in))		      \
	{								      \
	  printf ("getaddrinfo test %d: address size %zu, expected %zu\n",    \
		  no, (size_t) ai_res->ai_addrlen,			      \
		  sizeof (struct sockaddr_in));				      \
	  result = 1;							      \
	}								      \
      else if (strcmp (coraddr, \
		       inet_ntoa (((struct sockaddr_in *) ai_res->ai_addr)->sin_addr))\
	       != 0)							      \
	{								      \
	  printf ("getaddrinfo test %d: got value %s, expected %s\n",	      \
		  no,							      \
		  inet_ntoa (((struct sockaddr_in *) ai_res->ai_addr)->sin_addr), \
		  coraddr);						      \
	  result = 1;							      \
	}								      \
    }									      \
  else									      \
    {									      \
      char buf[100];							      \
									      \
      if (ai_res->ai_addrlen != sizeof (struct sockaddr_in6))		      \
	{								      \
	  printf ("getaddrinfo test %d: address size %zu, expected %zu\n",    \
		  no, (size_t) ai_res->ai_addrlen,			      \
		  sizeof (struct sockaddr_in6));			      \
	  result = 1;							      \
	}								      \
      else if (strcmp (coraddr, \
		       inet_ntop (AF_INET6,				      \
				  &((struct sockaddr_in6 *) ai_res->ai_addr)->sin6_addr,\
				  buf, sizeof (buf)))			      \
	       != 0)							      \
	{								      \
	  printf ("getaddrinfo test %d: got value %s, expected %s\n",	      \
		  no,							      \
		  inet_ntop (AF_INET6,					      \
			     & ((struct sockaddr_in6 *) ai_res->ai_addr)->sin6_addr, \
			     buf, sizeof (buf)),			      \
		  coraddr);						      \
	  result = 1;							      \
	}								      \
    }									      \
  if (ai_res != NULL && ai_res->ai_next != NULL)			      \
    {									      \
      puts ("expected only one result");				      \
      result = 1;							      \
    }									      \
  freeaddrinfo (ai_res)


  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  T (1, 0, "127.0.0.1", AF_INET, "127.0.0.1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  T (2, 0, "127.0.0.1", AF_INET, "127.0.0.1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_V4MAPPED;
  T (3, 0, "127.0.0.1", AF_INET6, "::ffff:127.0.0.1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;
  T (4, EAI_ADDRFAMILY, "127.0.0.1", AF_INET6, "");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  T (5, 0, "::1", AF_INET6, "::1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  T (6, EAI_ADDRFAMILY, "::1", AF_INET6, "");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;
  T (7, 0, "::1", AF_INET6, "::1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  T (8, 0, "::ffff:127.0.0.1", AF_INET6, "::ffff:127.0.0.1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  T (9, 0, "::ffff:127.0.0.1", AF_INET, "127.0.0.1");

  memset (&hints, '\0', sizeof (hints));
  hints.ai_family = AF_INET6;
  hints.ai_socktype = SOCK_STREAM;
  T (10, 0, "::ffff:127.0.0.1", AF_INET6, "::ffff:127.0.0.1");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
