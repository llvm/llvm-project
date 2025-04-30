/* Test case by Anders Carlsson <andersca@gnome.org>.  */
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>

int
main (void)
{
  struct addrinfo req, *ai;
  char name[] = "3ffe:0200:0064:0000:0202:b3ff:fe16:ddc5";

  memset (&req, '\0', sizeof req);
  req.ai_family = AF_INET6;

  /* This call used to crash.  We cannot expect the test machine to have
     IPv6 enabled so we just check that the call returns.  */
  getaddrinfo (name, NULL, &req, &ai);

  puts ("success!");
  return 0;
}
