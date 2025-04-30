/* Demux messages sent on the signal port.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <hurd/signal.h>
#include <stddef.h>

struct demux
  {
    struct demux *next;
    boolean_t (*demux) (mach_msg_header_t *inp,
			mach_msg_header_t *outp);
  };

struct demux *_hurd_msgport_demuxers = NULL;

extern boolean_t __msg_server (mach_msg_header_t *inp,
			       mach_msg_header_t *outp);

static boolean_t
msgport_server (mach_msg_header_t *inp,
		mach_msg_header_t *outp)
{
  extern boolean_t _S_msg_server (mach_msg_header_t *inp,
				  mach_msg_header_t *outp);
  extern boolean_t _S_exc_server (mach_msg_header_t *inp,
				  mach_msg_header_t *outp);
  struct demux *d;

  for (d = _hurd_msgport_demuxers; d != NULL; d = d->next)
    if ((*d->demux) (inp, outp))
      return 1;

  return (_S_exc_server (inp, outp)
	  || _S_msg_server (inp, outp));
}

/* This is the code that the signal thread runs.  */
void *
_hurd_msgport_receive (void *arg)
{
  /* Get our own sigstate cached so we never again have to take a lock to
     fetch it.  There is much code in hurdsig.c that operates with some
     sigstate lock held, which will deadlock with _hurd_thread_sigstate.

     Furthermore, in the pthread case this is the convenient spot
     to initialize _hurd_msgport_thread (see hurdsig.c:_hurdsig_init).  */

  _hurd_msgport_thread = _hurd_self_sigstate ()->thread;

  while (1)
    (void) __mach_msg_server (msgport_server, __vm_page_size, _hurd_msgport);
}
