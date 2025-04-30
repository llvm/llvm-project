/*
 * Copyright (c) 2010, Oracle America, Inc.
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
 *
 * This is the rpc server side idle loop
 * Wait for input, call server program.
 */

#include <errno.h>
#include <unistd.h>
#include <libintl.h>
#include <sys/poll.h>
#include <rpc/rpc.h>
#include <shlib-compat.h>

/* This function can be used as a signal handler to terminate the
   server loop.  */
void
svc_exit (void)
{
  free (svc_pollfd);
  svc_pollfd = NULL;
  svc_max_pollfd = 0;
}
libc_hidden_nolink_sunrpc (svc_exit, GLIBC_2_0)

void
svc_run (void)
{
  int i;
  struct pollfd *my_pollfd = NULL;
  int last_max_pollfd = 0;

  for (;;)
    {
      int max_pollfd = svc_max_pollfd;
      if (max_pollfd == 0 && svc_pollfd == NULL)
	break;

      if (last_max_pollfd != max_pollfd)
	{
	  struct pollfd *new_pollfd
	    = realloc (my_pollfd, sizeof (struct pollfd) * max_pollfd);

	  if (new_pollfd == NULL)
	    {
	      perror (_("svc_run: - out of memory"));
	      break;
	    }

	  my_pollfd = new_pollfd;
	  last_max_pollfd = max_pollfd;
	}

      for (i = 0; i < max_pollfd; ++i)
	{
	  my_pollfd[i].fd = svc_pollfd[i].fd;
	  my_pollfd[i].events = svc_pollfd[i].events;
	  my_pollfd[i].revents = 0;
	}

      switch (i = __poll (my_pollfd, max_pollfd, -1))
	{
	case -1:
	  if (errno == EINTR)
	    continue;
	  perror (_("svc_run: - poll failed"));
	  break;
	case 0:
	  continue;
	default:
	  svc_getreq_poll (my_pollfd, i);
	  continue;
	}
      break;
    }

  free (my_pollfd);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svc_run)
#else
libc_hidden_nolink_sunrpc (svc_run, GLIBC_2_0)
#endif
