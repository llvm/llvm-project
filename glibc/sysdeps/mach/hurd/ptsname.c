/* ptsname -- return the name of a pty slave given an FD to the pty master
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/term.h>


/* Return the pathname of the pseudo terminal slave associated with
   the master FD is open on, or NULL on errors.
   The returned storage is good until the next call to this function.  */
char *
ptsname (int fd)
{
  static string_t peername;
  error_t err;

  err = __ptsname_r (fd, peername, sizeof (peername));

  return err ? NULL : peername;
}


/* We don't need STP, but fill it for conformity with the Linux version...  */
int
__ptsname_internal (int fd, char *buf, size_t buflen, struct stat64 *stp)
{
  string_t peername;
  size_t len;
  error_t err;
  int ttype;

  if (HURD_DPORT_USE (fd, __term_get_bottom_type (port, &ttype)) == 0)
    {
      /* get_bottom_type suceeded, this is the slave side.  */
      errno = ENOTTY;
      return ENOTTY;
    }

  if (err = HURD_DPORT_USE (fd, __term_get_peername (port, peername)))
    {
      if (err == EMIG_BAD_ID || err == EOPNOTSUPP)
	err = ENOTTY;
      return __hurd_dfail (fd, err), errno;
    }

  len = __strnlen (peername, sizeof peername - 1) + 1;
  if (len > buflen)
    {
      errno = ERANGE;
      return ERANGE;
    }

  if (stp)
    {
      if (__stat64 (peername, stp) < 0)
	return errno;
    }

  memcpy (buf, peername, len);
  return 0;
}


/* Store at most BUFLEN characters of the pathname of the slave pseudo
   terminal associated with the master FD is open on in BUF.
   Return 0 on success, otherwise an error number.  */
int
__ptsname_r (int fd, char *buf, size_t buflen)
{
  return __ptsname_internal (fd, buf, buflen, NULL);
}
libc_hidden_def (__ptsname_r)
weak_alias (__ptsname_r, ptsname_r)
