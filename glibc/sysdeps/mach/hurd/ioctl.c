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
#include <sys/ioctl.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/signal.h>
#include <stdarg.h>
#include <mach/notify.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <hurd/ioctl.h>
#include <mach/mig_support.h>
#include <sysdep-cancel.h>

#include <hurd/ioctls.defs>

#define typesize(type)	(1 << (type))


/* Perform the I/O control operation specified by REQUEST on FD.
   The actual type and use of ARG and the return value depend on REQUEST.  */
int
__ioctl (int fd, unsigned long int request, ...)
{
#ifdef MACH_MSG_TYPE_CHAR
  /* Map individual type fields to Mach IPC types.  */
  static const int mach_types[] =
    { MACH_MSG_TYPE_CHAR, MACH_MSG_TYPE_INTEGER_16, MACH_MSG_TYPE_INTEGER_32,
      MACH_MSG_TYPE_INTEGER_64 };
#define io2mach_type(count, type) \
  ((mach_msg_type_t) { mach_types[type], typesize (type) * 8, count, 1, 0, 0 })
#endif

  /* Extract the type information encoded in the request.  */
  unsigned int type = _IOC_TYPE (request);

  /* Message buffer.  */
#define msg_align(x) \
  (((x) + sizeof (mach_msg_type_t) - 1) & ~(sizeof (mach_msg_type_t) - 1))
  struct
  {
#ifdef MACH_MSG_TYPE_BIT
    union
    {
      mig_reply_header_t header;
      struct
      {
	mach_msg_header_t	Head;
	int			RetCodeType;
	kern_return_t		RetCode;
      } header_typecheck;
    };
    char data[3 * sizeof (mach_msg_type_t)
	      + msg_align (_IOT_COUNT0 (type) * typesize (_IOT_TYPE0 (type)))
	      + msg_align (_IOT_COUNT1 (type) * typesize (_IOT_TYPE1 (type)))
	      + _IOT_COUNT2 (type) * typesize (_IOT_TYPE2 (type))];
#else  /* Untyped Mach IPC format.  */
    mig_reply_error_t header;
    char data[_IOT_COUNT0 (type) * typesize (_IOT_TYPE0 (type))
	      + _IOT_COUNT1 (type) * typesize (_IOT_TYPE1 (type))
	      + _IOT_COUNT2 (type) * typesize (_IOT_TYPE2 (type))];
    mach_msg_trailer_t trailer;
#endif
  } msg;
  mach_msg_header_t *const m = &msg.header.Head;
  mach_msg_id_t msgid;
  unsigned int reply_size;
#ifdef MACH_MSG_TYPE_BIT
  mach_msg_type_t *t;
#else
  void *p;
#endif

  void *arg = NULL;

  error_t err;

  /* Send the RPC already packed up in MSG to IOPORT
     and decode the return value.  */
  error_t send_rpc (io_t ioport)
    {
      error_t err;
#ifdef MACH_MSG_TYPE_BIT
      mach_msg_type_t *t = &msg.header.RetCodeType;
#else
      void *p = &msg.header.RetCode;
#endif

      /* Marshal the request arguments into the message buffer.
	 We must redo this work each time we retry the RPC after a SIGTTOU,
	 because the reply message containing the EBACKGROUND error code
	 clobbers the same message buffer also used for the request.  */

      if (_IOC_INOUT (request) & IOC_IN)
	{
	  /* We don't want to advance ARG since it will be used to copy out
	     too if IOC_OUT is also set.  */
	  void *argptr = arg;
	  int zero = 0;

	  if (request == TIOCFLUSH && !argptr)
	    argptr = &zero;

	  /* Pack an argument into the message buffer.  */
	  void in (unsigned int count, enum __ioctl_datum type)
	    {
	      if (count > 0)
		{
		  const size_t len = count * typesize ((unsigned int) type);
#ifdef MACH_MSG_TYPE_BIT
		  void *p = &t[1];
		  *t = io2mach_type (count, type);
		  p = __mempcpy (p, argptr, len);
		  p = (void *) (((uintptr_t) p + sizeof (*t) - 1)
				& ~(sizeof (*t) - 1));
		  t = p;
#else
		  p = __mempcpy (p, argptr, len);
#endif
		  argptr += len;
		}
	    }

	  /* Pack the argument data.  */
	  in (_IOT_COUNT0 (type), _IOT_TYPE0 (type));
	  in (_IOT_COUNT1 (type), _IOT_TYPE1 (type));
	  in (_IOT_COUNT2 (type), _IOT_TYPE2 (type));
	}
      else if (_IOC_INOUT (request) == IOC_VOID && _IOT_COUNT0 (type) != 0)
	{
	  /* The RPC takes a single integer_t argument.
	     Rather than pointing to the value, ARG is the value itself.  */
#ifdef MACH_MSG_TYPE_BIT
	  *t++ = io2mach_type (1, _IOTS (integer_t));
	  *(integer_t *) t = (integer_t) arg;
	  t = (void *) t + sizeof (integer_t);
#else
	  *(integer_t *) p = (integer_t) arg;
	  p = (void *) p + sizeof (integer_t);
#endif
	}

      memset (m, 0, sizeof *m);	/* Clear unused fields.  */
      m->msgh_size = (
#ifdef MACH_MSG_TYPE_BIT
		      (char *) t
#else
		      (char *) p
#endif
		      - (char *) &msg);
      m->msgh_remote_port = ioport;
      m->msgh_local_port = __mig_get_reply_port ();
      m->msgh_id = msgid;
      m->msgh_bits = MACH_MSGH_BITS (MACH_MSG_TYPE_COPY_SEND,
				     MACH_MSG_TYPE_MAKE_SEND_ONCE);
      err = _hurd_intr_rpc_mach_msg (m, MACH_SEND_MSG|MACH_RCV_MSG,
				     m->msgh_size, sizeof (msg),
				     m->msgh_local_port,
				     MACH_MSG_TIMEOUT_NONE,
				     MACH_PORT_NULL);
      switch (err)
	{
	case MACH_MSG_SUCCESS:
	  break;
	case MACH_SEND_INVALID_REPLY:
	case MACH_RCV_INVALID_NAME:
	  __mig_dealloc_reply_port (m->msgh_local_port);
	  /* Fall through.  */
	default:
	  return err;
	}

      if ((m->msgh_bits & MACH_MSGH_BITS_COMPLEX))
	{
	  /* Allow no ports or VM.  */
	  __mach_msg_destroy (m);
	  /* Want to return a different error below for a different msgid.  */
	  if (m->msgh_id == msgid + 100)
	    return MIG_TYPE_ERROR;
	}

      if (m->msgh_id != msgid + 100)
	return (m->msgh_id == MACH_NOTIFY_SEND_ONCE
		? MIG_SERVER_DIED : MIG_REPLY_MISMATCH);

      if (m->msgh_size != reply_size
	  && m->msgh_size != sizeof msg.header)
	return MIG_TYPE_ERROR;

#ifdef MACH_MSG_TYPE_BIT
      if (msg.header_typecheck.RetCodeType
	  != ((union { mach_msg_type_t t; int i; })
	    { t: io2mach_type (1, _IOTS (msg.header.RetCode)) }).i)
	return MIG_TYPE_ERROR;
#endif
      return msg.header.RetCode;
    }

  if (_IOT_COUNT0 (type) != 0)
    {
      /* Data need either be sent, received, or even both.  */
      va_list ap;

      va_start (ap, request);
      arg = va_arg (ap, void *);
      va_end (ap);
    }

  {
    /* Check for a registered handler for REQUEST.  */
    ioctl_handler_t handler = _hurd_lookup_ioctl_handler (request);
    if (handler)
      {
	/* This handler groks REQUEST.  Se lo puntamonos.  */
	int save = errno;
	int result = (*handler) (fd, request, arg);
	if (result != -1 || errno != ENOTTY)
	  return result;

	/* The handler doesn't really grok this one.
	   Try the normal RPC translation.  */
	errno = save;
      }
  }

  /* Compute the Mach message ID for the RPC from the group and command
     parts of the ioctl request.  */
  msgid = IOC_MSGID (request);

  /* Compute the expected size of the reply.  There is a standard header
     consisting of the message header and the reply code.  Then, for out
     and in/out ioctls, there come the data with their type headers.  */
  reply_size = sizeof msg.header;

  if (_IOC_INOUT (request) & IOC_OUT)
    {
      inline void figure_reply (unsigned int count, enum __ioctl_datum type)
	{
	  if (count > 0)
	    {
#ifdef MACH_MSG_TYPE_BIT
	      /* Add the size of the type and data.  */
	      reply_size += sizeof (mach_msg_type_t) + typesize (type) * count;
	      /* Align it to word size.  */
	      reply_size += sizeof (mach_msg_type_t) - 1;
	      reply_size &= ~(sizeof (mach_msg_type_t) - 1);
#else
	      reply_size += typesize (type) * count;
#endif
	    }
	}
      figure_reply (_IOT_COUNT0 (type), _IOT_TYPE0 (type));
      figure_reply (_IOT_COUNT1 (type), _IOT_TYPE1 (type));
      figure_reply (_IOT_COUNT2 (type), _IOT_TYPE2 (type));
    }

  /* Marshal the arguments into the request message and make the RPC.
     This wrapper function handles EBACKGROUND returns, turning them
     into either SIGTTOU or EIO.  */
  if (request == TIOCDRAIN)
    {
      /* This is a cancellation point.  */
      int cancel_oldtype = LIBC_CANCEL_ASYNC();
      err = HURD_DPORT_USE_CANCEL (fd, _hurd_ctty_output (port, ctty, send_rpc));
      LIBC_CANCEL_RESET (cancel_oldtype);
    }
  else
    err = HURD_DPORT_USE (fd, _hurd_ctty_output (port, ctty, send_rpc));

#ifdef MACH_MSG_TYPE_BIT
  t = (mach_msg_type_t *) msg.data;
#else
  p = (void *) msg.data;
#endif
  switch (err)
    {
      /* Unpack the message buffer into the argument location.  */
      int out (unsigned int count, unsigned int type,
	       void *store, void **update)
	{
	  if (count > 0)
	    {
	      const size_t len = count * typesize (type);
#ifdef MACH_MSG_TYPE_BIT
	      union { mach_msg_type_t t; int i; } ipctype;
	      ipctype.t = io2mach_type (count, type);
	      if (*(int *) t != ipctype.i)
		return 1;
	      ++t;
	      memcpy (store, t, len);
	      if (update != NULL)
		*update += len;
	      t = (void *) (((uintptr_t) t + len + sizeof (*t) - 1)
			    & ~(sizeof (*t) - 1));
#else
	      memcpy (store, p, len);
	      p += len;
	      if (update != NULL)
		*update += len;
#endif
	    }
	  return 0;
	}

    case 0:
      if (m->msgh_size != reply_size
	  || ((_IOC_INOUT (request) & IOC_OUT)
	      && (out (_IOT_COUNT0 (type), _IOT_TYPE0 (type), arg, &arg)
		  || out (_IOT_COUNT1 (type), _IOT_TYPE1 (type), arg, &arg)
		  || out (_IOT_COUNT2 (type), _IOT_TYPE2 (type), arg, &arg))))
	return __hurd_fail (MIG_TYPE_ERROR);
      return 0;

    case MIG_BAD_ID:
    case EOPNOTSUPP:
      /* The server didn't understand the RPC.  */
      err = ENOTTY;
      /* Fall through.  */
    default:
      return __hurd_fail (err);
    }
}

libc_hidden_def (__ioctl)
weak_alias (__ioctl, ioctl)
