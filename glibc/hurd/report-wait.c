/* Report on what a thread in our task is waiting for.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <hurd/fd.h>
#include <string.h>
#include <assert.h>
#include <hurd/msg_server.h>
#include <thread_state.h>
#include <intr-msg.h>

static char *
describe_number (char *description, const char *flavor, long int i, size_t size)
{
  unsigned long int j;
  char *limit = description + size;
  char *p = flavor == NULL ? description : __stpncpy (description, flavor, size);
  char *end;

  if (p == limit)
    return p;

  /* Handle sign.  */
  if (i < 0)
    {
      i = -i;
      *p++ = '-';
    }

  if (p == limit)
    return p;

  /* Allocate space for the number at the end of DESCRIPTION.  */
  for (j = i; j >= 10; j /= 10)
    p++;
  end = p + 1;

  if (end < limit)
    *end = '\0';
  else
    end = limit;

  do
    {
      if (p < limit)
	*p = '0' + i % 10;
      p--;
      i /= 10;
    } while (i != 0);

  return end;
}

static char *
describe_port (char *description, mach_port_t port, size_t size)
{
  int i;

  if (port == MACH_PORT_NULL)
    return __stpncpy (description, "(null)", size);
  if (port == MACH_PORT_DEAD)
    return __stpncpy (description, "(dead)", size);

  if (port == __mach_task_self ())
    return __stpncpy (description, "task-self", size);

  for (i = 0; i < _hurd_nports; ++i)
    if (port == _hurd_ports[i].port)
      return describe_number (description, "init#", i, size);

  if (_hurd_init_dtable)
    {
      for (i = 0; i < _hurd_init_dtablesize; ++i)
	if (port == _hurd_init_dtable[i])
	  return describe_number (description, "fd#", i, size);
    }
  if (_hurd_dtable)
    {
      for (i = 0; i < _hurd_dtablesize; ++i)
	if (_hurd_dtable[i] == NULL)
	  continue;
	else if (port == _hurd_dtable[i]->port.port)
	  return describe_number (description, "fd#", i, size);
	else if (port == _hurd_dtable[i]->ctty.port)
	  return describe_number (description, "bgfd#", i, size);
    }

  return describe_number (description, "port#", port, size);
}


/* We want _HURD_ITIMER_THREAD, but don't want to link in the itimer code
   unnecessarily.  */
#if 0 /* libc.so.0.0 needs this defined, so make it a weak alias for now.  */
extern thread_t _hurd_itimer_thread; /* XXX */
weak_extern (_hurd_itimer_thread)
#else
static thread_t default_hurd_itimer_thread;
weak_alias (default_hurd_itimer_thread, _hurd_itimer_thread)
#endif

kern_return_t
_S_msg_report_wait (mach_port_t msgport, thread_t thread,
		    string_t description, mach_msg_id_t *msgid)
{
  char *limit = description + 1024; /* XXX */
  char *cur = description;
  *msgid = 0;

  if (thread == _hurd_msgport_thread)
    /* Cute.  */
    cur = __stpncpy (cur, "msgport", limit - cur);
  else if (&_hurd_itimer_thread && thread == _hurd_itimer_thread)
    cur = __stpncpy (cur, "itimer", limit - cur);
  else
    {
      /* Make sure this is really one of our threads.  */

      struct hurd_sigstate *ss;

      __mutex_lock (&_hurd_siglock);
      for (ss = _hurd_sigstates; ss != NULL; ss = ss->next)
	if (ss->thread == thread)
	  break;
      __mutex_unlock (&_hurd_siglock);
      if (ss == NULL)
	/* To hell with you.  */
	return EINVAL;

      if (ss->suspended != MACH_PORT_NULL)
	cur = __stpncpy (cur, "sigsuspend", limit - cur);
      else
	{
	  /* Examine the thread's state to see if it is blocked in an RPC.  */

	  struct machine_thread_state state;
	  mach_msg_type_number_t count = MACHINE_THREAD_STATE_COUNT;
	  error_t err;

	  err = __thread_get_state (thread, MACHINE_THREAD_STATE_FLAVOR,
				    (natural_t *) &state, &count);
	  if (err)
	    return err;
	  assert (count == MACHINE_THREAD_STATE_COUNT);
	  if (SYSCALL_EXAMINE (&state, msgid))
	    {
	      mach_port_t send_port, rcv_port;
	      mach_msg_option_t option;
	      mach_msg_timeout_t timeout;

	      /* Blocked in a system call.  */
	      if (*msgid == -25
		  /* mach_msg system call.  Examine its parameters.  */
		  && MSG_EXAMINE (&state, msgid, &rcv_port, &send_port,
				  &option, &timeout) == 0)
		{
		  if (send_port != MACH_PORT_NULL && *msgid != 0)
		    {
		      /* For the normal case of RPCs, we consider the
			 destination port to be the interesting thing
			 whether we are in fact sending or receiving at the
			 moment.  That tells us who we are waiting for the
			 reply from.  */
		      if (send_port == ss->intr_port)
			{
			  /* This is a Hurd interruptible RPC.
			     Mark it by surrounding the port description
			     string with [...] brackets.  */
			  if (cur < limit)
			    *cur++ = '[';
			  cur = describe_port (cur, send_port, limit - cur);
			  if (cur < limit)
			    *cur++ = ']';
			}
		      else
			cur = describe_port (cur, send_port, limit - cur);
		    }
		  else if (rcv_port != MACH_PORT_NULL)
		    {
		      /* This system call had no send port, but had a
			 receive port.  The msgid we extracted is then just
			 some garbage or perhaps the msgid of the last
			 message this thread received, but it's not a
			 helpful thing to return.  */
		      cur = describe_port (cur, rcv_port, limit - cur);
		      cur = __stpncpy (cur, ":rcv", limit - cur);
		      *msgid = 0;
		    }
		  else if ((option & (MACH_RCV_MSG|MACH_RCV_TIMEOUT))
			   == (MACH_RCV_MSG|MACH_RCV_TIMEOUT))
		    {
		      /* A receive with no valid port can be used for a
			 pure timeout.  Report the timeout value (counted
			 in milliseconds); note this is the original total
			 time, not the time remaining.  */
		      cur = describe_number (cur, 0, timeout, limit - cur);
		      cur = __stpncpy (cur, "ms", limit - cur);
		      *msgid = 0;
		    }
		  else
		    {
		      cur = __stpncpy (cur, "mach_msg", limit - cur);
		      *msgid = 0;
		    }
		}
	      else		/* Some other system call.  */
		{
		  cur = describe_number (cur, "syscall#", *msgid, limit - cur);
		  *msgid = 0;
		}
	    }
	}
    }

  __mach_port_deallocate (__mach_task_self (), thread);

  if (cur == limit)
    return ENOMEM;

  *cur = '\0';
  return 0;
}

kern_return_t
_S_msg_describe_ports (mach_port_t msgport, mach_port_t refport,
		       mach_port_t *ports, mach_msg_type_number_t nports,
		       char **desc, mach_msg_type_number_t *desclen)
{
  char *p, *end;

  if (__USEPORT (AUTH, msgport != port))
    return EPERM;

  end = *desc + *desclen;
  p = *desc;
  while (nports-- > 0)
    {
      char this[200];
      describe_port (this, *ports++, sizeof this);
      p = __stpncpy (p, this, end - p);
      if (p == end && p[-1] != '\0')
	return ENOMEM;
    }

  *desclen = p - *desc;
  return 0;
}
