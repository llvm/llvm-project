/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <alloca.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <netdb.h>
#include <not-cancel.h>
#include <stdbool.h>

#define HOSTIDFILE "/etc/hostid"

#ifdef SET_PROCEDURE
int
sethostid (long int id)
{
  int fd;
  ssize_t written;
  int32_t id32 = id;

  /* Test for appropriate rights to set host ID.  */
  if (__libc_enable_secure)
    {
      __set_errno (EPERM);
      return -1;
    }

  /* Make sure the ID is not too large.  Needed for bi-arch support.   */
  if (id32 != id)
    {
      __set_errno (EOVERFLOW);
      return -1;
    }

  /* Open file for writing.  Everybody is allowed to read this file.  */
  fd = __open_nocancel (HOSTIDFILE, O_CREAT|O_WRONLY|O_TRUNC, 0644);
  if (fd < 0)
    return -1;

  written = __write_nocancel (fd, &id32, sizeof (id32));

  __close_nocancel_nostatus (fd);

  return written != sizeof (id32) ? -1 : 0;
}

#else
# include <string.h>
# include <sys/param.h>
# include <resolv/netdb.h>
# include <netinet/in.h>
# include <scratch_buffer.h>

long int
gethostid (void)
{
  char hostname[MAXHOSTNAMELEN + 1];
  struct hostent hostbuf, *hp;
  int32_t id;
  struct in_addr in;
  int herr;
  int fd;

  /* First try to get the ID from a former invocation of sethostid.  */
  fd = __open_nocancel (HOSTIDFILE, O_RDONLY|O_LARGEFILE, 0);
  if (fd >= 0)
    {
      ssize_t n = __read_nocancel (fd, &id, sizeof (id));

      __close_nocancel_nostatus (fd);

      if (n == sizeof (id))
	return id;
    }

  /* Getting from the file was not successful.  An intelligent guess
     for a unique number of a host is its IP address.  To get the IP
     address we need to know the host name.  */
  if (__gethostname (hostname, MAXHOSTNAMELEN) < 0 || hostname[0] == '\0')
    /* This also fails.  Return and arbitrary value.  */
    return 0;

  /* Determine the IP address of the host name.  */
  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);
  while (true)
    {
      int ret = __gethostbyname_r (hostname, &hostbuf,
				   tmpbuf.data, tmpbuf.length, &hp, &herr);
      if (ret == 0 && hp != NULL)
	break;
      else
	{
	  /* Enlarge the buffer on ERANGE.  */
	  if (ret != 0 && herr == NETDB_INTERNAL && errno == ERANGE)
	    {
	      if (!scratch_buffer_grow (&tmpbuf))
		return 0;
	    }
	  /* Other errors are a failure.  Return an arbitrary value.  */
	  else
	    {
	      scratch_buffer_free (&tmpbuf);
	      return 0;
	    }
	}
    }

  in.s_addr = 0;
  memcpy (&in, hp->h_addr,
	  (int) sizeof (in) < hp->h_length ? (int) sizeof (in) : hp->h_length);
  scratch_buffer_free (&tmpbuf);
  /* For the return value to be not exactly the IP address we do some
     bit fiddling.  */
  return (int32_t) (in.s_addr << 16 | in.s_addr >> 16);
}
#endif
