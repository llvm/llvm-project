/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libintl.h>
#include <tls-internal.h>

/* Return a string describing the meaning of the signal number SIGNUM.  */
char *
strsignal (int signum)
{
  const char *desc = __sigdescr_np (signum);
  if (desc != NULL)
    return _(desc);

  struct tls_internal_t *tls_internal = __glibc_tls_internal ();
  free (tls_internal->strsignal_buf);

  int r;
#ifdef SIGRTMIN
  if (signum >= SIGRTMIN && signum <= SIGRTMAX)
    r = __asprintf (&tls_internal->strsignal_buf, _("Real-time signal %d"),
		    signum - SIGRTMIN);
  else
#endif
    r = __asprintf (&tls_internal->strsignal_buf, _("Unknown signal %d"),
		    signum);

  if (r == -1)
    tls_internal->strsignal_buf = NULL;

  return tls_internal->strsignal_buf;
}
