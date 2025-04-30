/*
 * get_myaddress.c
 *
 * Get client's IP address via ioctl.  This avoids using the yellowpages.
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

#include <rpc/types.h>
#include <rpc/clnt.h>
#include <rpc/pmap_prot.h>
#include <sys/socket.h>
#include <stdio.h>
#include <unistd.h>
#include <libintl.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <shlib-compat.h>

/*
 * don't use gethostbyname, which would invoke yellow pages
 *
 * Avoid loopback interfaces.  We return information from a loopback
 * interface only if there are no other possible interfaces.
 */
void
get_myaddress (struct sockaddr_in *addr)
{
  struct ifaddrs *ifa;

  if (getifaddrs (&ifa) != 0)
    {
      perror ("get_myaddress: getifaddrs");
      exit (1);
    }

  int loopback = 0;
  struct ifaddrs *run;

 again:
  run = ifa;
  while (run != NULL)
    {
      if ((run->ifa_flags & IFF_UP)
	  && run->ifa_addr != NULL
	  && run->ifa_addr->sa_family == AF_INET
	  && (!(run->ifa_flags & IFF_LOOPBACK)
	      || (loopback == 1 && (run->ifa_flags & IFF_LOOPBACK))))
	{
	  *addr = *((struct sockaddr_in *) run->ifa_addr);
	  addr->sin_port = htons (PMAPPORT);
	  goto out;
	}

      run = run->ifa_next;
    }

  if (loopback == 0)
    {
      loopback = 1;
      goto again;
    }
 out:
  freeifaddrs (ifa);

  /* The function is horribly specified.  It does not return any error
     if no interface is up.  Probably this won't happen (at least
     loopback is there) but still...  */
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (get_myaddress)
#else
libc_hidden_nolink_sunrpc (get_myaddress, GLIBC_2_0)
#endif
