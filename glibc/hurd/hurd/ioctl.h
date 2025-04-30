/* User-registered handlers for specific `ioctl' requests.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef	_HURD_IOCTL_H
#define	_HURD_IOCTL_H	1

#define	__need___va_list
#include <stdarg.h>
#include <bits/ioctls.h>
#include <mach/port.h>


/* Type of handler function, called like ioctl to do its entire job.  */
typedef int (*ioctl_handler_t) (int fd, int request, void *arg);

/* Structure that records an ioctl handler.  */
struct ioctl_handler
  {
    /* Range of handled _IOC_NOTYPE (REQUEST) values.  */
    int first_request, last_request;

    /* Handler function, called like ioctl to do its entire job.  */
    ioctl_handler_t handler;

    struct ioctl_handler *next;	/* Next handler.  */
  };


/* Register HANDLER to handle ioctls with REQUEST values between
   FIRST_REQUEST and LAST_REQUEST inclusive.  Returns zero if successful.
   Return nonzero and sets `errno' for an error.  */

extern int hurd_register_ioctl_handler (int first_request, int last_request,
					ioctl_handler_t handler);


/* Define a library-internal handler for ioctl commands between FIRST and
   LAST inclusive.  The last element gratuitously references HANDLER to
   avoid `defined but not used' warnings.  */

#define	_HURD_HANDLE_IOCTLS_1(handler, first, last, moniker)		      \
  static const struct ioctl_handler handler##_ioctl_handler##moniker	      \
	__attribute__ ((__unused__)) =					      \
    { _IOC_NOTYPE (first), _IOC_NOTYPE (last),				      \
	(ioctl_handler_t) (handler), NULL };				      \
  text_set_element (_hurd_ioctl_handler_lists,				      \
                    handler##_ioctl_handler##moniker)
#define	_HURD_HANDLE_IOCTLS(handler, first, last)			      \
  _HURD_HANDLE_IOCTLS_1 (handler, first, last, first##_to_##last)

/* Define a library-internal handler for a single ioctl command.  */

#define _HURD_HANDLE_IOCTL(handler, ioctl) \
  _HURD_HANDLE_IOCTLS_1 (handler, ioctl, ioctl, ioctl##_only)


/* Install a new CTTYID port, atomically updating the dtable appropriately.
   This consumes the send right passed in.  */

void _hurd_locked_install_cttyid (mach_port_t cttyid);

/* Lookup the handler for the given ioctl request.  */

ioctl_handler_t _hurd_lookup_ioctl_handler (int request);


#endif	/* hurd/ioctl.h */
