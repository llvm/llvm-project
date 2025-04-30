/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#include <pwd.h>
#include <unistd.h>
#include <not-cancel.h>
#include <scratch_buffer.h>

#define STATIC static
static int getlogin_r_fd0 (char *name, size_t namesize);
#define __getlogin_r getlogin_r_fd0
#include <sysdeps/unix/getlogin_r.c>
#undef __getlogin_r


/* Try to determine login name from /proc/self/loginuid and return 0
   if successful.  If /proc/self/loginuid cannot be read return -1.
   Otherwise return the error number.  */

int
attribute_hidden
__getlogin_r_loginuid (char *name, size_t namesize)
{
  int fd = __open_nocancel ("/proc/self/loginuid", O_RDONLY);
  if (fd == -1)
    return -1;

  /* We are reading a 32-bit number.  12 bytes are enough for the text
     representation.  If not, something is wrong.  */
  char uidbuf[12];
  ssize_t n = TEMP_FAILURE_RETRY (__read_nocancel (fd, uidbuf,
						   sizeof (uidbuf)));
  __close_nocancel_nostatus (fd);

  uid_t uid;
  char *endp;
  if (n <= 0
      || n == sizeof (uidbuf)
      || (uidbuf[n] = '\0',
	  uid = strtoul (uidbuf, &endp, 10),
	  endp == uidbuf || *endp != '\0'))
    return -1;

  /* If there is no login uid, linux sets /proc/self/loginid to the sentinel
     value of, (uid_t) -1, so check if that value is set and return early to
     avoid making unneeded nss lookups. */
  if (uid == (uid_t) -1)
    {
      __set_errno (ENXIO);
      return ENXIO;
    }

  struct passwd pwd;
  struct passwd *tpwd;
  int result = 0;
  int res;
  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  while ((res =  __getpwuid_r (uid, &pwd,
			       tmpbuf.data, tmpbuf.length, &tpwd)) == ERANGE)
    {
      if (!scratch_buffer_grow (&tmpbuf))
	{
	  result = ENOMEM;
	  goto out;
	}
    }

  if (res != 0 || tpwd == NULL)
    {
      result = -1;
      goto out;
    }

  size_t needed = strlen (pwd.pw_name) + 1;
  if (needed > namesize)
    {
      __set_errno (ERANGE);
      result = ERANGE;
      goto out;
    }

  memcpy (name, pwd.pw_name, needed);

 out:
  scratch_buffer_free (&tmpbuf);
  return result;
}


/* Return at most NAME_LEN characters of the login name of the user in NAME.
   If it cannot be determined or some other error occurred, return the error
   code.  Otherwise return 0.  */

int
__getlogin_r (char *name, size_t namesize)
{
  int res = __getlogin_r_loginuid (name, namesize);
  if (res >= 0)
    return res;

  return getlogin_r_fd0 (name, namesize);
}
libc_hidden_def (__getlogin_r)
weak_alias (__getlogin_r, getlogin_r)
libc_hidden_weak (getlogin_r)
