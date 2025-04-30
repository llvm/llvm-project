/*-
 * Copyright (c) 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)login_tty.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */

#include <errno.h>
#include <sys/param.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <utmp.h>
#include <shlib-compat.h>

int
__login_tty (int fd)
{
	__setsid();
#ifdef TIOCSCTTY
	if (__ioctl(fd, TIOCSCTTY, NULL) == -1)
		return (-1);
#else
	{
	  /* This might work.  */
	  char *fdname = ttyname (fd);
	  int newfd;
	  if (fdname)
	    {
	      if (fd != 0)
		_close (0);
	      if (fd != 1)
		__close (1);
	      if (fd != 2)
		__close (2);
	      newfd = __open64 (fdname, O_RDWR);
	      __close (newfd);
	    }
	}
#endif
	while (__dup2(fd, 0) == -1 && errno == EBUSY)
	  ;
	while (__dup2(fd, 1) == -1 && errno == EBUSY)
	  ;
	while (__dup2(fd, 2) == -1 && errno == EBUSY)
	  ;
	if (fd > 2)
		__close(fd);
	return (0);
}
versioned_symbol (libc, __login_tty, login_tty, GLIBC_2_34);
libc_hidden_ver (__login_tty, login_tty)

#if OTHER_SHLIB_COMPAT (libutil, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libutil, __login_tty, login_tty, GLIBC_2_0);
#endif
