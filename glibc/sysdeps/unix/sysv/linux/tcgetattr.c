/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sysdep.h>

/* The difference here is that the termios structure used in the
   kernel is not the same as we use in the libc.  Therefore we must
   translate it here.  */
#include <kernel_termios.h>

/* Put the state of FD into *TERMIOS_P.  */
int
__tcgetattr (int fd, struct termios *termios_p)
{
  struct __kernel_termios k_termios;
  int retval;

  retval = INLINE_SYSCALL (ioctl, 3, fd, TCGETS, &k_termios);

  if (__glibc_likely (retval == 0))
    {
      termios_p->c_iflag = k_termios.c_iflag;
      termios_p->c_oflag = k_termios.c_oflag;
      termios_p->c_cflag = k_termios.c_cflag;
      termios_p->c_lflag = k_termios.c_lflag;
      termios_p->c_line = k_termios.c_line;
#if _HAVE_STRUCT_TERMIOS_C_ISPEED
# if _HAVE_C_ISPEED
      termios_p->c_ispeed = k_termios.c_ispeed;
# else
      termios_p->c_ispeed = k_termios.c_cflag & (CBAUD | CBAUDEX);
# endif
#endif
#if _HAVE_STRUCT_TERMIOS_C_OSPEED
# if _HAVE_C_OSPEED
      termios_p->c_ospeed = k_termios.c_ospeed;
# else
      termios_p->c_ospeed = k_termios.c_cflag & (CBAUD | CBAUDEX);
# endif
#endif
      if (sizeof (cc_t) == 1 || _POSIX_VDISABLE == 0
	  || (unsigned char) _POSIX_VDISABLE == (unsigned char) -1)
	memset (__mempcpy (&termios_p->c_cc[0], &k_termios.c_cc[0],
			   __KERNEL_NCCS * sizeof (cc_t)),
		_POSIX_VDISABLE, (NCCS - __KERNEL_NCCS) * sizeof (cc_t));
      else
	{
	  memcpy (&termios_p->c_cc[0], &k_termios.c_cc[0],
		  __KERNEL_NCCS * sizeof (cc_t));

	  for (size_t cnt = __KERNEL_NCCS; cnt < NCCS; ++cnt)
	    termios_p->c_cc[cnt] = _POSIX_VDISABLE;
	}
    }

  return retval;
}

libc_hidden_def (__tcgetattr)
weak_alias (__tcgetattr, tcgetattr)
