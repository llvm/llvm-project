/* Return point to next ancillary data entry in message header.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <stddef.h>
#include <sys/socket.h>


struct cmsghdr *
__cmsg_nxthdr (struct msghdr *mhdr, struct cmsghdr *cmsg)
{
  if ((size_t) cmsg->cmsg_len < sizeof (struct cmsghdr))
    /* The kernel header does this so there may be a reason.  */
    return NULL;

  cmsg = (struct cmsghdr *) ((unsigned char *) cmsg
			     + CMSG_ALIGN (cmsg->cmsg_len));
  if ((unsigned char *) (cmsg + 1) > ((unsigned char *) mhdr->msg_control
				      + mhdr->msg_controllen)
      || ((unsigned char *) cmsg + CMSG_ALIGN (cmsg->cmsg_len)
	  > ((unsigned char *) mhdr->msg_control + mhdr->msg_controllen)))
    /* No more entries.  */
    return NULL;
  return cmsg;
}
libc_hidden_def (__cmsg_nxthdr)
