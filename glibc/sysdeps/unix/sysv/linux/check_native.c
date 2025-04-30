/* Determine whether interfaces use native transport.  Linux version.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <assert.h>
#include <errno.h>
#include <ifaddrs.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <sys/ioctl.h>

#include <asm/types.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

#include <not-cancel.h>

#include "netlinkaccess.h"

void
__check_native (uint32_t a1_index, int *a1_native,
		uint32_t a2_index, int *a2_native)
{
  int fd = __socket (PF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE);

  struct sockaddr_nl nladdr;
  memset (&nladdr, '\0', sizeof (nladdr));
  nladdr.nl_family = AF_NETLINK;

  socklen_t addr_len = sizeof (nladdr);
  bool use_malloc = false;

  if (fd < 0)
    return;

  if (__bind (fd, (struct sockaddr *) &nladdr, sizeof (nladdr)) != 0
      || __getsockname (fd, (struct sockaddr *) &nladdr, &addr_len) != 0)
    goto out;

  pid_t pid = nladdr.nl_pid;
  struct req
  {
    struct nlmsghdr nlh;
    struct rtgenmsg g;
    /* struct rtgenmsg consists of a single byte.  This means there
       are three bytes of padding included in the REQ definition.
       We make them explicit here.  */
    char pad[3];
  } req;

  req.nlh.nlmsg_len = sizeof (req);
  req.nlh.nlmsg_type = RTM_GETLINK;
  req.nlh.nlmsg_flags = NLM_F_ROOT | NLM_F_MATCH | NLM_F_REQUEST;
  req.nlh.nlmsg_pid = 0;
  req.nlh.nlmsg_seq = time_now ();
  req.g.rtgen_family = AF_UNSPEC;

  assert (sizeof (req) - offsetof (struct req, pad) == 3);
  memset (req.pad, '\0', sizeof (req.pad));

  memset (&nladdr, '\0', sizeof (nladdr));
  nladdr.nl_family = AF_NETLINK;

#ifdef PAGE_SIZE
  /* Help the compiler optimize out the malloc call if PAGE_SIZE
     is constant and smaller or equal to PTHREAD_STACK_MIN/4.  */
  const size_t buf_size = PAGE_SIZE;
#else
  const size_t buf_size = __getpagesize ();
#endif
  char *buf;

  if (__libc_use_alloca (buf_size))
    buf = alloca (buf_size);
  else
    {
      buf = malloc (buf_size);
      if (buf != NULL)
	use_malloc = true;
      else
	goto out;
    }

  struct iovec iov = { buf, buf_size };

  if (TEMP_FAILURE_RETRY (__sendto (fd, (void *) &req, sizeof (req), 0,
				    (struct sockaddr *) &nladdr,
				    sizeof (nladdr))) < 0)
    goto out;

  bool done = false;
  do
    {
      struct msghdr msg =
	{
	  .msg_name = (void *) &nladdr,
	  .msg_namelen =  sizeof (nladdr),
	  .msg_iov = &iov,
	  .msg_iovlen = 1,
	  .msg_control = NULL,
	  .msg_controllen = 0,
	  .msg_flags = 0
	};

      ssize_t read_len = TEMP_FAILURE_RETRY (__recvmsg (fd, &msg, 0));
      __netlink_assert_response (fd, read_len);
      if (read_len < 0)
	goto out;

      if (msg.msg_flags & MSG_TRUNC)
	goto out;

      struct nlmsghdr *nlmh;
      for (nlmh = (struct nlmsghdr *) buf;
	   NLMSG_OK (nlmh, (size_t) read_len);
	   nlmh = (struct nlmsghdr *) NLMSG_NEXT (nlmh, read_len))
	{
	  if (nladdr.nl_pid != 0 || (pid_t) nlmh->nlmsg_pid != pid
	      || nlmh->nlmsg_seq != req.nlh.nlmsg_seq)
	    continue;

	  if (nlmh->nlmsg_type == RTM_NEWLINK)
	    {
	      struct ifinfomsg *ifim = (struct ifinfomsg *) NLMSG_DATA (nlmh);
	      int native = (ifim->ifi_type != ARPHRD_TUNNEL6
			    && ifim->ifi_type != ARPHRD_TUNNEL
			    && ifim->ifi_type != ARPHRD_SIT);

	      if (a1_index == ifim->ifi_index)
		{
		  *a1_native = native;
		  a1_index = 0xffffffffu;
		}
	      if (a2_index == ifim->ifi_index)
		{
		  *a2_native = native;
		  a2_index = 0xffffffffu;
		}

	      if (a1_index == 0xffffffffu
		  && a2_index == 0xffffffffu)
		goto out;
	    }
	  else if (nlmh->nlmsg_type == NLMSG_DONE)
	    /* We found the end, leave the loop.  */
	    done = true;
	}
    }
  while (! done);

out:
  __close_nocancel_nostatus (fd);

  if (use_malloc)
    free (buf);
}
