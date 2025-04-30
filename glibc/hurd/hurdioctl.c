/* ioctl commands which must be done in the C library.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <hurd/fd.h>
#include <sys/ioctl.h>
#include <hurd/ioctl.h>
#include <string.h>


/* Symbol set of ioctl handler lists.  If there are user-registered
   handlers, one of these lists will contain them.  The other lists are
   handlers built into the library.  */
symbol_set_define (_hurd_ioctl_handler_lists)

/* Look up REQUEST in the set of handlers.  */
ioctl_handler_t
_hurd_lookup_ioctl_handler (int request)
{
  void *const *ptr;
  const struct ioctl_handler *h;

  /* Mask off the type bits, so that we see requests in a single group as a
     contiguous block of values.  */
  request = _IOC_NOTYPE (request);

  for (ptr = symbol_set_first_element (_hurd_ioctl_handler_lists);
       !symbol_set_end_p (_hurd_ioctl_handler_lists, ptr);
       ++ptr)
    for (h = *ptr; h != NULL; h = h->next)
      if (request >= h->first_request && request <= h->last_request)
	return h->handler;

  return NULL;
}

#include <fcntl.h>

/* Find out how many bytes may be read from FD without blocking.  */

static int
fioctl (int fd,
	int request,
	int *arg)
{
  error_t err;

  *(volatile int *) arg = *arg;

  switch (request)
    {
    default:
      err = ENOTTY;
      break;

    case FIONREAD:
      {
	mach_msg_type_number_t navail;
	err = HURD_DPORT_USE (fd, __io_readable (port, &navail));
	if (!err)
	  *arg = (int) navail;
      }
      break;

    case FIONBIO:
      err = HURD_DPORT_USE (fd, (*arg
				 ? __io_set_some_openmodes
				 : __io_clear_some_openmodes)
			    (port, O_NONBLOCK));
      break;

    case FIOASYNC:
      err = HURD_DPORT_USE (fd, (*arg
				 ? __io_set_some_openmodes
				 : __io_clear_some_openmodes)
			    (port, O_ASYNC));
      break;

    case FIOSETOWN:
      err = HURD_DPORT_USE (fd, __io_mod_owner (port, *arg));
      break;

    case FIOGETOWN:
      err = HURD_DPORT_USE (fd, __io_get_owner (port, arg));
      break;
    }

  return err ? __hurd_dfail (fd, err) : 0;
}

_HURD_HANDLE_IOCTLS (fioctl, FIOGETOWN, FIONREAD);


static int
fioclex (int fd,
	 int request)
{
  int flag;

  switch (request)
    {
    default:
      return __hurd_fail (ENOTTY);
    case FIOCLEX:
      flag = FD_CLOEXEC;
      break;
    case FIONCLEX:
      flag = 0;
      break;
    }

  return __fcntl (fd, F_SETFD, flag);
}
_HURD_HANDLE_IOCTLS (fioclex, FIOCLEX, FIONCLEX);

#include <hurd/term.h>
#include <hurd/tioctl.h>

/* Install a new CTTYID port, atomically updating the dtable appropriately.
   This consumes the send right passed in.  */

void
_hurd_locked_install_cttyid (mach_port_t cttyid)
{
  mach_port_t old;
  struct hurd_port *const port = &_hurd_ports[INIT_PORT_CTTYID];
  struct hurd_userlink ulink;
  int i;

  /* Install the new cttyid port, and preserve it with a ulink.
     We unroll the _hurd_port_set + _hurd_port_get here so that
     there is no window where the cell is unlocked and CTTYID could
     be changed by another thread.  (We also delay the deallocation
     of the old port until the end, to minimize the duration of the
     critical section.)

     It is important that changing the cttyid port is only ever done by
     holding the dtable lock continuously while updating the port cell and
     re-ctty'ing the dtable; dtable.c assumes we do this.  Otherwise, the
     pgrp-change notification code in dtable.c has to worry about racing
     against us here in odd situations.  The one exception to this is
     setsid, which holds the dtable lock while changing the pgrp and
     clearing the cttyid port, and then unlocks the dtable lock to allow


  */

  __spin_lock (&port->lock);
  old = _hurd_userlink_clear (&port->users) ? port->port : MACH_PORT_NULL;
  port->port = cttyid;
  cttyid = _hurd_port_locked_get (port, &ulink);

  for (i = 0; i < _hurd_dtablesize; ++i)
    {
      struct hurd_fd *const d = _hurd_dtable[i];
      mach_port_t newctty = MACH_PORT_NULL;

      if (d == NULL)
	/* Nothing to do for an unused descriptor cell.  */
	continue;

      if (cttyid != MACH_PORT_NULL)
	/* We do have some controlling tty.  */
	HURD_PORT_USE (&d->port,
		       ({ mach_port_t id;
			  /* Get the io object's cttyid port.  */
			  if (! __term_getctty (port, &id))
			    {
			      if (id == cttyid /* Is it ours?  */
				  /* Get the ctty io port.  */
				  && __term_open_ctty (port,
						       _hurd_pid, _hurd_pgrp,
						       &newctty))
				/* XXX it is our ctty but the call failed? */
				newctty = MACH_PORT_NULL;
			      __mach_port_deallocate (__mach_task_self (), id);
			    }
			  0;
			}));

      /* Install the new ctty port.  */
      _hurd_port_set (&d->ctty, newctty);
    }

  __mutex_unlock (&_hurd_dtable_lock);

  if (old != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), old);
  _hurd_port_free (port, &ulink, cttyid);
}

static void
install_ctty (mach_port_t cttyid)
{
  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_dtable_lock);
  _hurd_locked_install_cttyid (cttyid);
  HURD_CRITICAL_END;
}


/* Called when we have received a message saying to use a new ctty ID port.  */

error_t
_hurd_setcttyid (mach_port_t cttyid)
{
  error_t err;

  if (cttyid != MACH_PORT_NULL)
    {
      /* Give the new send right a user reference.
	 This is a good way to check that it is valid.  */
      if (err = __mach_port_mod_refs (__mach_task_self (), cttyid,
				      MACH_PORT_RIGHT_SEND, 1))
	return err;
    }

  /* Install the port, consuming the reference we just created.  */
  install_ctty (cttyid);

  return 0;
}


static inline error_t
do_tiocsctty (io_t port, io_t ctty)
{
  mach_port_t cttyid;
  error_t err;

  if (ctty != MACH_PORT_NULL)
    /* PORT is already the ctty.  Nothing to do.  */
    return 0;

  /* Get PORT's cttyid port.  */
  err = __term_getctty (port, &cttyid);
  if (err)
    return err;

  /* Change the terminal's pgrp to ours.  */
  err = __tioctl_tiocspgrp (port, _hurd_pgrp);
  if (err)
    __mach_port_deallocate (__mach_task_self (), cttyid);
  else
    /* Make it our own.  */
    install_ctty (cttyid);

  return err;
}

/* Make FD be the controlling terminal.
   This function is called for `ioctl (fd, TCIOSCTTY)'.  */

static int
tiocsctty (int fd,
	   int request)		/* Always TIOCSCTTY.  */
{
  return __hurd_fail (HURD_DPORT_USE (fd, do_tiocsctty (port, ctty)));
}
_HURD_HANDLE_IOCTL (tiocsctty, TIOCSCTTY);

/* Dissociate from the controlling terminal.  */

static int
tiocnotty (int fd,
	   int request)		/* Always TIOCNOTTY.  */
{
  mach_port_t fd_cttyid;
  error_t err;

  if (err = HURD_DPORT_USE (fd, __term_getctty (port, &fd_cttyid)))
    return __hurd_fail (err);

  if (__USEPORT (CTTYID, port != fd_cttyid))
    err = EINVAL;

  __mach_port_deallocate (__mach_task_self (), fd_cttyid);

  if (err)
    return __hurd_fail (err);

  /* Clear our cttyid port.  */
  install_ctty (MACH_PORT_NULL);

  return 0;
}
_HURD_HANDLE_IOCTL (tiocnotty, TIOCNOTTY);

#include <hurd/pfinet.h>
#include <net/if.h>
#include <netinet/in.h>

/* Fill in the buffer IFC->IFC_BUF of length IFC->IFC_LEN with a list
   of ifr structures, one for each network interface.  */
static int
siocgifconf (int fd, int request, struct ifconf *ifc)
{
  error_t err;
  size_t data_len = ifc->ifc_len;
  char *data = ifc->ifc_buf;

  if (data_len <= 0)
    return 0;

  err = HURD_DPORT_USE (fd, __pfinet_siocgifconf (port, ifc->ifc_len,
						  &data, &data_len));
  if (data_len < ifc->ifc_len)
    ifc->ifc_len = data_len;
  if (data != ifc->ifc_buf)
    {
      memcpy (ifc->ifc_buf, data, ifc->ifc_len);
      __vm_deallocate (__mach_task_self (), (vm_address_t) data, data_len);
    }
  return err ? __hurd_dfail (fd, err) : 0;
}
_HURD_HANDLE_IOCTL (siocgifconf, SIOCGIFCONF);
