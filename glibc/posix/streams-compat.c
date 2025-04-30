/* Compatibility symbols for the unimplemented XSI STREAMS extension.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_30)

# include <errno.h>
# include <fcntl.h>

struct strbuf;

int
attribute_compat_text_section
fattach (int fildes, const char *path)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, fattach, fattach, GLIBC_2_1);

int
attribute_compat_text_section
fdetach (const char *path)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, fdetach, fdetach, GLIBC_2_1);


int
attribute_compat_text_section
getmsg (int fildes, struct strbuf *ctlptr, struct strbuf *dataptr, int *flagsp)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, getmsg, getmsg, GLIBC_2_1);

int
attribute_compat_text_section
getpmsg (int fildes, struct strbuf *ctlptr, struct strbuf *dataptr, int *bandp,
	 int *flagsp)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, getpmsg, getpmsg, GLIBC_2_1);

int
attribute_compat_text_section
isastream (int fildes)
{
  /* In general we do not have a STREAMS implementation and therefore
     return 0.  But for invalid file descriptors we have to return an
     error.  */
  if (__fcntl (fildes, F_GETFD) < 0)
    return -1;

  /* No STREAM.  */
  return 0;
}
compat_symbol (libc, isastream, isastream, GLIBC_2_1);

int
attribute_compat_text_section
putmsg (int fildes, const struct strbuf *ctlptr, const struct strbuf *dataptr,
	int flags)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, putmsg, putmsg, GLIBC_2_1);

int
attribute_compat_text_section
putpmsg (int fildes, const struct strbuf *ctlptr, const struct strbuf *dataptr,
	 int band, int flags)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, putpmsg, putpmsg, GLIBC_2_1);

#endif /* SHLIB_COMPAT */
