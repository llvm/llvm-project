/* Guts of both `select' and `poll' for Hurd.
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

#include <sys/time.h>
#include <sys/types.h>
#include <sys/poll.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/io_request.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>
#include <sysdep-cancel.h>

/* All user select types.  */
#define SELECT_ALL (SELECT_READ | SELECT_WRITE | SELECT_URG)

/* Used to record that a particular select rpc returned.  Must be distinct
   from SELECT_ALL (which better not have the high bit set).  */
#define SELECT_RETURNED ((SELECT_ALL << 1) & ~SELECT_ALL)
#define SELECT_ERROR (SELECT_RETURNED << 1)

/* Check the first NFDS descriptors either in POLLFDS (if nonnnull) or in
   each of READFDS, WRITEFDS, EXCEPTFDS that is nonnull.  If TIMEOUT is not
   NULL, time out after waiting the interval specified therein.  Returns
   the number of ready descriptors, or -1 for errors.  */
int
_hurd_select (int nfds,
	      struct pollfd *pollfds,
	      fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
	      const struct timespec *timeout, const sigset_t *sigmask)
{
  int i;
  mach_port_t portset, sigport;
  int got, ready;
  error_t err;
  fd_set rfds, wfds, xfds;
  int firstfd, lastfd;
  mach_msg_id_t reply_msgid;
  mach_msg_timeout_t to;
  struct timespec ts;
  struct
    {
      struct hurd_userlink ulink;
      struct hurd_fd *cell;
      mach_port_t io_port;
      int type;
      mach_port_t reply_port;
      int error;
    } d[nfds];
  sigset_t oset;
  struct hurd_sigstate *ss;

  union typeword		/* Use this to avoid unkosher casts.  */
    {
      mach_msg_type_t type;
      uint32_t word;
    };
  assert (sizeof (union typeword) == sizeof (mach_msg_type_t));
  assert (sizeof (uint32_t) == sizeof (mach_msg_type_t));

  if (nfds < 0 || (pollfds == NULL && nfds > FD_SETSIZE))
    {
      errno = EINVAL;
      return -1;
    }

#define IO_SELECT_REPLY_MSGID (21012 + 100) /* XXX */
#define IO_SELECT_TIMEOUT_REPLY_MSGID (21031 + 100) /* XXX */

  if (timeout == NULL)
    reply_msgid = IO_SELECT_REPLY_MSGID;
  else
    {
      struct timespec now;

      if (timeout->tv_sec < 0 || ! valid_nanoseconds (timeout->tv_nsec))
	{
	  errno = EINVAL;
	  return -1;
	}

      err = __clock_gettime (CLOCK_REALTIME, &now);
      if (err)
	return -1;

      ts.tv_sec = now.tv_sec + timeout->tv_sec;
      ts.tv_nsec = now.tv_nsec + timeout->tv_nsec;

      if (ts.tv_nsec >= 1000000000)
	{
	  ts.tv_sec++;
	  ts.tv_nsec -= 1000000000;
	}

      if (ts.tv_sec < 0)
	ts.tv_sec = LONG_MAX; /* XXX */

      reply_msgid = IO_SELECT_TIMEOUT_REPLY_MSGID;
    }

  if (sigmask)
    {
      /* Add a port to the portset for the case when we get the signal even
         before calling __mach_msg.  */

      sigport = __mach_reply_port ();

      ss = _hurd_self_sigstate ();
      _hurd_sigstate_lock (ss);
      /* And tell the signal thread to message us when a signal arrives.  */
      ss->suspended = sigport;
      _hurd_sigstate_unlock (ss);

      if (__sigprocmask (SIG_SETMASK, sigmask, &oset))
	{
	  _hurd_sigstate_lock (ss);
	  ss->suspended = MACH_PORT_NULL;
	  _hurd_sigstate_unlock (ss);
	  __mach_port_destroy (__mach_task_self (), sigport);
	  return -1;
	}
    }
  else
    sigport = MACH_PORT_NULL;

  if (pollfds)
    {
      int error = 0;
      /* Collect interesting descriptors from the user's `pollfd' array.
	 We do a first pass that reads the user's array before taking
	 any locks.  The second pass then only touches our own stack,
	 and gets the port references.  */

      for (i = 0; i < nfds; ++i)
	if (pollfds[i].fd >= 0)
	  {
	    int type = 0;
	    if (pollfds[i].events & POLLIN)
	      type |= SELECT_READ;
	    if (pollfds[i].events & POLLOUT)
	      type |= SELECT_WRITE;
	    if (pollfds[i].events & POLLPRI)
	      type |= SELECT_URG;

	    d[i].io_port = pollfds[i].fd;
	    d[i].type = type;
	  }
	else
	  d[i].type = 0;

      HURD_CRITICAL_BEGIN;
      __mutex_lock (&_hurd_dtable_lock);

      for (i = 0; i < nfds; ++i)
	if (d[i].type != 0)
	  {
	    const int fd = (int) d[i].io_port;

	    if (fd < _hurd_dtablesize)
	      {
		d[i].cell = _hurd_dtable[fd];
		if (d[i].cell != NULL)
		  {
		    d[i].io_port = _hurd_port_get (&d[i].cell->port,
						   &d[i].ulink);
		    if (d[i].io_port != MACH_PORT_NULL)
		      continue;
		  }
	      }

	    /* Bogus descriptor, make it EBADF already.  */
	    d[i].error = EBADF;
	    d[i].type = SELECT_ERROR;
	    error = 1;
	  }

      __mutex_unlock (&_hurd_dtable_lock);
      HURD_CRITICAL_END;

      if (error)
	{
	  /* Set timeout to 0.  */
	  err = __clock_gettime (CLOCK_REALTIME, &ts);
	  if (err)
	    {
	      /* Really bad luck.  */
	      err = errno;
	      HURD_CRITICAL_BEGIN;
	      __mutex_lock (&_hurd_dtable_lock);
	      while (i-- > 0)
		if (d[i].type & ~SELECT_ERROR != 0)
		  _hurd_port_free (&d[i].cell->port, &d[i].ulink,
				   d[i].io_port);
	      __mutex_unlock (&_hurd_dtable_lock);
	      HURD_CRITICAL_END;
	      if (sigmask)
		__sigprocmask (SIG_SETMASK, &oset, NULL);
	      errno = err;
	      return -1;
	    }
	  reply_msgid = IO_SELECT_TIMEOUT_REPLY_MSGID;
	}

      lastfd = i - 1;
      firstfd = i == 0 ? lastfd : 0;
    }
  else
    {
      /* Collect interested descriptors from the user's fd_set arguments.
	 Use local copies so we can't crash from user bogosity.  */

      if (readfds == NULL)
	FD_ZERO (&rfds);
      else
	rfds = *readfds;
      if (writefds == NULL)
	FD_ZERO (&wfds);
      else
	wfds = *writefds;
      if (exceptfds == NULL)
	FD_ZERO (&xfds);
      else
	xfds = *exceptfds;

      HURD_CRITICAL_BEGIN;
      __mutex_lock (&_hurd_dtable_lock);

      /* Collect the ports for interesting FDs.  */
      firstfd = lastfd = -1;
      for (i = 0; i < nfds; ++i)
	{
	  int type = 0;
	  if (readfds != NULL && FD_ISSET (i, &rfds))
	    type |= SELECT_READ;
	  if (writefds != NULL && FD_ISSET (i, &wfds))
	    type |= SELECT_WRITE;
	  if (exceptfds != NULL && FD_ISSET (i, &xfds))
	    type |= SELECT_URG;
	  d[i].type = type;
	  if (type)
	    {
	      if (i < _hurd_dtablesize)
		{
		  d[i].cell = _hurd_dtable[i];
		  if (d[i].cell != NULL)
		    d[i].io_port = _hurd_port_get (&d[i].cell->port,
						   &d[i].ulink);
		}
	      if (i >= _hurd_dtablesize || d[i].cell == NULL ||
		  d[i].io_port == MACH_PORT_NULL)
		{
		  /* If one descriptor is bogus, we fail completely.  */
		  while (i-- > 0)
		    if (d[i].type != 0)
		      _hurd_port_free (&d[i].cell->port, &d[i].ulink,
				       d[i].io_port);
		  break;
		}
	      lastfd = i;
	      if (firstfd == -1)
		firstfd = i;
	    }
	}

      __mutex_unlock (&_hurd_dtable_lock);
      HURD_CRITICAL_END;

      if (i < nfds)
	{
	  if (sigmask)
	    __sigprocmask (SIG_SETMASK, &oset, NULL);
	  errno = EBADF;
	  return -1;
	}

      if (nfds > _hurd_dtablesize)
	nfds = _hurd_dtablesize;
    }


  err = 0;
  got = 0;

  /* Send them all io_select request messages.  */

  if (firstfd == -1)
    {
      if (sigport == MACH_PORT_NULL)
	/* But not if there were no ports to deal with at all.
	   We are just a pure timeout.  */
	portset = __mach_reply_port ();
      else
	portset = sigport;
    }
  else
    {
      portset = MACH_PORT_NULL;

      for (i = firstfd; i <= lastfd; ++i)
	if (!(d[i].type & ~SELECT_ERROR))
	  d[i].reply_port = MACH_PORT_NULL;
	else
	  {
	    int type = d[i].type;
	    d[i].reply_port = __mach_reply_port ();
	    if (timeout == NULL)
	      err = __io_select_request (d[i].io_port, d[i].reply_port, type);
	    else
	      err = __io_select_timeout_request (d[i].io_port, d[i].reply_port,
						 ts, type);
	    if (!err)
	      {
		if (firstfd == lastfd && sigport == MACH_PORT_NULL)
		  /* When there's a single descriptor, we don't need a
		     portset, so just pretend we have one, but really
		     use the single reply port.  */
		  portset = d[i].reply_port;
		else if (got == 0)
		  /* We've got multiple reply ports, so we need a port set to
		     multiplex them.  */
		  {
		    /* We will wait again for a reply later.  */
		    if (portset == MACH_PORT_NULL)
		      /* Create the portset to receive all the replies on.  */
		      err = __mach_port_allocate (__mach_task_self (),
						  MACH_PORT_RIGHT_PORT_SET,
						  &portset);
		    if (! err)
		      /* Put this reply port in the port set.  */
		      __mach_port_move_member (__mach_task_self (),
					       d[i].reply_port, portset);
		  }
	      }
	    else
	      {
		/* No error should happen, but record it for later
		   processing.  */
		d[i].error = err;
		d[i].type |= SELECT_ERROR;
		++got;
	      }
	    _hurd_port_free (&d[i].cell->port, &d[i].ulink, d[i].io_port);
	  }

      if (got == 0 && sigport != MACH_PORT_NULL)
	{
	  if (portset == MACH_PORT_NULL)
	    /* Create the portset to receive the signal message on.  */
	    __mach_port_allocate (__mach_task_self (), MACH_PORT_RIGHT_PORT_SET,
				  &portset);
	  /* Put the signal reply port in the port set.  */
	  __mach_port_move_member (__mach_task_self (), sigport, portset);
	}
    }

  /* GOT is the number of replies (or errors), while READY is the number of
     replies with at least one type bit set.  */
  ready = 0;

  /* Now wait for reply messages.  */
  if (!err && got == 0)
    {
      /* Now wait for io_select_reply messages on PORT,
	 timing out as appropriate.  */

      union
	{
	  mach_msg_header_t head;
#ifdef MACH_MSG_TRAILER_MINIMUM_SIZE
	  struct
	    {
	      mach_msg_header_t head;
	      NDR_record_t ndr;
	      error_t err;
	    } error;
	  struct
	    {
	      mach_msg_header_t head;
	      NDR_record_t ndr;
	      error_t err;
	      int result;
	      mach_msg_trailer_t trailer;
	    } success;
#else
	  struct
	    {
	      mach_msg_header_t head;
	      union typeword err_type;
	      error_t err;
	    } error;
	  struct
	    {
	      mach_msg_header_t head;
	      union typeword err_type;
	      error_t err;
	      union typeword result_type;
	      int result;
	    } success;
#endif
	} msg;
      mach_msg_option_t options;
      error_t msgerr;

      /* We rely on servers to implement the timeout, but when there are none,
	 do it on the client side.  */
      if (timeout != NULL && firstfd == -1)
	{
	  options = MACH_RCV_TIMEOUT;
	  to = timeout->tv_sec * 1000 + (timeout->tv_nsec + 999999) / 1000000;
	}
      else
	{
	  options = 0;
	  to = MACH_MSG_TIMEOUT_NONE;
	}

      int cancel_oldtype = LIBC_CANCEL_ASYNC();
      while ((msgerr = __mach_msg (&msg.head,
				   MACH_RCV_MSG | MACH_RCV_INTERRUPT | options,
				   0, sizeof msg, portset, to,
				   MACH_PORT_NULL)) == MACH_MSG_SUCCESS)
	{
	  LIBC_CANCEL_RESET (cancel_oldtype);

	  /* We got a message.  Decode it.  */
#ifdef MACH_MSG_TYPE_BIT
	  const union typeword inttype =
	  { type:
	    { MACH_MSG_TYPE_INTEGER_T, sizeof (integer_t) * 8, 1, 1, 0, 0 }
	  };
#endif

	  if (sigport != MACH_PORT_NULL && sigport == msg.head.msgh_local_port)
	    {
	      /* We actually got interrupted by a signal before
		 __mach_msg; poll for further responses and then
		 return quickly. */
	      err = EINTR;
	      goto poll;
	    }

	  if (msg.head.msgh_id == reply_msgid
	      && msg.head.msgh_size >= sizeof msg.error
	      && !(msg.head.msgh_bits & MACH_MSGH_BITS_COMPLEX)
#ifdef MACH_MSG_TYPE_BIT
	      && msg.error.err_type.word == inttype.word
#endif
	      )
	    {
	      /* This is a properly formatted message so far.
		 See if it is a success or a failure.  */
	      if (msg.error.err == EINTR
		  && msg.head.msgh_size == sizeof msg.error)
		{
		  /* EINTR response; poll for further responses
		     and then return quickly.  */
		  err = EINTR;
		  goto poll;
		}
	      /* Keep in mind msg.success.result can be 0 if a timeout
		 occurred.  */
	      if (msg.error.err
#ifdef MACH_MSG_TYPE_BIT
		  || msg.success.result_type.word != inttype.word
#endif
		  || msg.head.msgh_size != sizeof msg.success)
		{
		  /* Error or bogus reply.  */
		  if (!msg.error.err)
		    msg.error.err = EIO;
		  __mach_msg_destroy (&msg.head);
		}

	      /* Look up the respondent's reply port and record its
		 readiness.  */
	      {
		int had = got;
		if (firstfd != -1)
		  for (i = firstfd; i <= lastfd; ++i)
		    if (d[i].type
			&& d[i].reply_port == msg.head.msgh_local_port)
		      {
			if (msg.error.err)
			  {
			    d[i].error = msg.error.err;
			    d[i].type = SELECT_ERROR;
			    ++ready;
			  }
			else
			  {
			    d[i].type &= msg.success.result;
			    if (d[i].type)
			      ++ready;
			  }

			d[i].type |= SELECT_RETURNED;
			++got;
		      }
		assert (got > had);
	      }
	    }

	  if (msg.head.msgh_remote_port != MACH_PORT_NULL)
	    __mach_port_deallocate (__mach_task_self (),
				    msg.head.msgh_remote_port);

	  if (got)
	  poll:
	    {
	      /* Poll for another message.  */
	      to = 0;
	      options |= MACH_RCV_TIMEOUT;
	    }
	}
      LIBC_CANCEL_RESET (cancel_oldtype);

      if (msgerr == MACH_RCV_INTERRUPTED)
	/* Interruption on our side (e.g. signal reception).  */
	err = EINTR;

      if (ready)
	/* At least one descriptor is known to be ready now, so we will
	   return success.  */
	err = 0;
    }

  if (firstfd != -1)
    for (i = firstfd; i <= lastfd; ++i)
      if (d[i].reply_port != MACH_PORT_NULL)
	__mach_port_destroy (__mach_task_self (), d[i].reply_port);

  if (sigport != MACH_PORT_NULL)
    {
      _hurd_sigstate_lock (ss);
      ss->suspended = MACH_PORT_NULL;
      _hurd_sigstate_unlock (ss);
      __mach_port_destroy (__mach_task_self (), sigport);
    }

  if ((firstfd == -1 && sigport == MACH_PORT_NULL)
      || ((firstfd != lastfd || sigport != MACH_PORT_NULL) && portset != MACH_PORT_NULL))
    /* Destroy PORTSET, but only if it's not actually the reply port for a
       single descriptor (in which case it's destroyed in the previous loop;
       not doing it here is just a bit more efficient).  */
    __mach_port_destroy (__mach_task_self (), portset);

  if (err)
    {
      if (sigmask)
	__sigprocmask (SIG_SETMASK, &oset, NULL);
      return __hurd_fail (err);
    }

  if (pollfds)
    /* Fill in the `revents' members of the user's array.  */
    for (i = 0; i < nfds; ++i)
      {
	int type = d[i].type;
	int_fast16_t revents = 0;

	if (type & SELECT_ERROR)
	  switch (d[i].error)
	    {
	      case EPIPE:
		revents = POLLHUP;
		break;
	      case EBADF:
		revents = POLLNVAL;
		break;
	      default:
		revents = POLLERR;
		break;
	    }
	else
	  if (type & SELECT_RETURNED)
	    {
	      if (type & SELECT_READ)
		revents |= POLLIN;
	      if (type & SELECT_WRITE)
		revents |= POLLOUT;
	      if (type & SELECT_URG)
		revents |= POLLPRI;
	    }

	pollfds[i].revents = revents;
      }
  else
    {
      /* Below we recalculate READY to include an increment for each operation
	 allowed on each fd.  */
      ready = 0;

      /* Set the user bitarrays.  We only ever have to clear bits, as all
	 desired ones are initially set.  */
      if (firstfd != -1)
	for (i = firstfd; i <= lastfd; ++i)
	  {
	    int type = d[i].type;

	    if ((type & SELECT_RETURNED) == 0)
	      type = 0;

	    /* Callers of select don't expect to see errors, so we simulate
	       readiness of the erring object and the next call hopefully
	       will get the error again.  */
	    if (type & SELECT_ERROR)
	      {
		type = 0;
		if (readfds != NULL && FD_ISSET (i, readfds))
		  type |= SELECT_READ;
		if (writefds != NULL && FD_ISSET (i, writefds))
		  type |= SELECT_WRITE;
		if (exceptfds != NULL && FD_ISSET (i, exceptfds))
		  type |= SELECT_URG;
	      }

	    if (type & SELECT_READ)
	      ready++;
	    else if (readfds)
	      FD_CLR (i, readfds);
	    if (type & SELECT_WRITE)
	      ready++;
	    else if (writefds)
	      FD_CLR (i, writefds);
	    if (type & SELECT_URG)
	      ready++;
	    else if (exceptfds)
	      FD_CLR (i, exceptfds);
	  }
    }

  if (sigmask && __sigprocmask (SIG_SETMASK, &oset, NULL))
    return -1;

  return ready;
}
