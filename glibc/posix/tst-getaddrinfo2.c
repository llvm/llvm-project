/* Test by David L Stevens <dlstevens@us.ibm.com> [BZ #358] */
#include <errno.h>
#include <netdb.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

static int
do_test (void)
{
  const char portstr[] = "583";
  int port = atoi (portstr);
  struct addrinfo hints, *aires, *pai;
  int rv;
  int res = 1;

  memset (&hints, 0, sizeof (hints));
  hints.ai_family = AF_INET;
  rv = getaddrinfo (NULL, portstr, &hints, &aires);
  if (rv == 0)
    {
      struct sockaddr_in *psin = 0;
      int got_tcp, got_udp;
      int err = 0;

      got_tcp = got_udp = 0;
      for (pai = aires; pai; pai = pai->ai_next)
        {
          printf ("ai_family=%d, ai_addrlen=%d, ai_socktype=%d",
                  (int) pai->ai_family, (int) pai->ai_addrlen,
                  (int) pai->ai_socktype);
          if (pai->ai_family == AF_INET)
            printf (", port=%d",
                    ntohs (((struct sockaddr_in *) pai->ai_addr)->sin_port));
          puts ("");

          err |= pai->ai_family != AF_INET;
          err |= pai->ai_addrlen != sizeof (struct sockaddr_in);
          err |= pai->ai_addr == 0;
          if (pai->ai_family == AF_INET)
            err |=
              ntohs (((struct sockaddr_in *) pai->ai_addr)->sin_port) != port;
          got_tcp |= pai->ai_socktype == SOCK_STREAM;
          got_udp |= pai->ai_socktype == SOCK_DGRAM;
          if (err)
            break;
        }
      if (err)
        {
          printf ("FAIL getaddrinfo IPv4 socktype 0,513: "
                  "fam %d alen %d addr %p addr/fam %d "
                  "addr/port %d H[%d]\n",
                  pai->ai_family, pai->ai_addrlen, psin,
                  psin ? psin->sin_family : 0,
                  psin ? psin->sin_port : 0,
                  psin ? htons (psin->sin_port) : 0);
        }
      else if (got_tcp && got_udp)
        {
          printf ("SUCCESS getaddrinfo IPv4 socktype 0,513\n");
          res = 0;
        }
      else
        printf ("FAIL getaddrinfo IPv4 socktype 0,513 TCP %d"
                " UDP %d\n", got_tcp, got_udp);
      freeaddrinfo (aires);
    }
  else
    printf ("FAIL getaddrinfo IPv4 socktype 0,513 returns %d "
            "(\"%s\")\n", rv, gai_strerror (rv));

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
