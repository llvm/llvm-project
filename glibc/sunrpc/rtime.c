/*
 * Copyright (c) 2010, Oracle America, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the "Oracle America, Inc." nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * rtime - get time from remote machine
 *
 * gets time, obtaining value from host
 * on the udp/time socket.  Since timeserver returns
 * with time of day in seconds since Jan 1, 1900,  must
 * subtract seconds before Jan 1, 1970 to get
 * what unix uses.
 */
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <rpc/rpc.h>
#include <rpc/clnt.h>
#include <sys/types.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <rpc/auth_des.h>
#include <errno.h>
#include <netinet/in.h>
#include <shlib-compat.h>

#define NYEARS	(u_long)(1970 - 1900)
#define TOFFSET (u_long)(60*60*24*(365*NYEARS + (NYEARS/4)))

static void do_close (int);

static void
do_close (int s)
{
  int save;

  save = errno;
  __close (s);
  __set_errno (save);
}

int
rtime (struct sockaddr_in *addrp, struct rpc_timeval *timep,
       struct rpc_timeval *timeout)
{
  int s;
  struct pollfd fd;
  int milliseconds;
  int res;
  /* RFC 868 says the time is transmitted as a 32-bit value.  */
  uint32_t thetime;
  struct sockaddr_in from;
  socklen_t fromlen;
  int type;

  if (timeout == NULL)
    type = SOCK_STREAM;
  else
    type = SOCK_DGRAM;

  s = __socket (AF_INET, type, 0);
  if (s < 0)
    return (-1);

  addrp->sin_family = AF_INET;
  addrp->sin_port = htons (IPPORT_TIMESERVER);
  if (type == SOCK_DGRAM)
    {
      res = __sendto (s, (char *) &thetime, sizeof (thetime), 0,
		      (struct sockaddr *) addrp, sizeof (*addrp));
      if (res < 0)
	{
	  do_close (s);
	  return -1;
	}
      milliseconds = (timeout->tv_sec * 1000) + (timeout->tv_usec / 1000);
      fd.fd = s;
      fd.events = POLLIN;
      do
	res = __poll (&fd, 1, milliseconds);
      while (res < 0 && errno == EINTR);
      if (res <= 0)
	{
	  if (res == 0)
	    __set_errno (ETIMEDOUT);
	  do_close (s);
	  return (-1);
	}
      fromlen = sizeof (from);
      res = __recvfrom (s, (char *) &thetime, sizeof (thetime), 0,
			(struct sockaddr *) &from, &fromlen);
      do_close (s);
      if (res < 0)
	return -1;
    }
  else
    {
      if (__connect (s, (struct sockaddr *) addrp, sizeof (*addrp)) < 0)
	{
	  do_close (s);
	  return -1;
	}
      res = __read (s, (char *) &thetime, sizeof (thetime));
      do_close (s);
      if (res < 0)
	return (-1);
    }
  if (res != sizeof (thetime))
    {
      __set_errno (EIO);
      return -1;
    }
  thetime = ntohl (thetime);
  timep->tv_sec = thetime - TOFFSET;
  timep->tv_usec = 0;
  return 0;
}
libc_hidden_nolink_sunrpc (rtime, GLIBC_2_1)
