/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989, 1995 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
#include <mach/port.h>
#include <mach/message.h>
#include <mach.h>

#ifdef MACH_MSG_OVERWRITE
/* In variants with this feature, the actual system call is
   __mach_msg_overwrite_trap.  */
mach_msg_return_t
__mach_msg_trap (mach_msg_header_t *msg,
		 mach_msg_option_t option,
		 mach_msg_size_t send_size,
		 mach_msg_size_t rcv_size,
		 mach_port_t rcv_name,
		 mach_msg_timeout_t timeout,
		 mach_port_t notify)
{
  return __mach_msg_overwrite_trap (msg, option, send_size,
				    rcv_size, rcv_name, timeout, notify,
				    MACH_MSG_NULL, 0);
}
weak_alias (__mach_msg_trap, mach_msg_trap)

/* See comments below in __mach_msg.  */
mach_msg_return_t
__mach_msg_overwrite (mach_msg_header_t *msg,
		      mach_msg_option_t option,
		      mach_msg_size_t send_size,
		      mach_msg_size_t rcv_size,
		      mach_port_t rcv_name,
		      mach_msg_timeout_t timeout,
		      mach_port_t notify,
		      mach_msg_header_t *rcv_msg,
		      mach_msg_size_t rcv_msg_size)

{
  mach_msg_return_t ret;

  /* Consider the following cases:
     1. Errors in pseudo-receive (eg, MACH_SEND_INTERRUPTED
     plus special bits).
     2. Use of MACH_SEND_INTERRUPT/MACH_RCV_INTERRUPT options.
     3. RPC calls with interruptions in one/both halves.
  */

  ret = __mach_msg_overwrite_trap (msg, option, send_size,
				   rcv_size, rcv_name, timeout, notify,
				   rcv_msg, rcv_msg_size);
  if (ret == MACH_MSG_SUCCESS)
    return MACH_MSG_SUCCESS;

  if (!(option & MACH_SEND_INTERRUPT))
    while (ret == MACH_SEND_INTERRUPTED)
      ret = __mach_msg_overwrite_trap (msg, option, send_size,
				       rcv_size, rcv_name, timeout, notify,
				       rcv_msg, rcv_msg_size);

  if (!(option & MACH_RCV_INTERRUPT))
    while (ret == MACH_RCV_INTERRUPTED)
      ret = __mach_msg_overwrite_trap (msg, option & ~MACH_SEND_MSG,
				       0, rcv_size, rcv_name, timeout, notify,
				       rcv_msg, rcv_msg_size);

  return ret;
}
weak_alias (__mach_msg_overwrite, mach_msg_overwrite)
#endif

mach_msg_return_t
__mach_msg (mach_msg_header_t *msg,
	    mach_msg_option_t option,
	    mach_msg_size_t send_size,
	    mach_msg_size_t rcv_size,
	    mach_port_t rcv_name,
	    mach_msg_timeout_t timeout,
	    mach_port_t notify)
{
  mach_msg_return_t ret;

  /* Consider the following cases:
     1. Errors in pseudo-receive (eg, MACH_SEND_INTERRUPTED
     plus special bits).
     2. Use of MACH_SEND_INTERRUPT/MACH_RCV_INTERRUPT options.
     3. RPC calls with interruptions in one/both halves.
     */

  ret = __mach_msg_trap (msg, option, send_size,
			 rcv_size, rcv_name, timeout, notify);
  if (ret == MACH_MSG_SUCCESS)
    return MACH_MSG_SUCCESS;

  if (!(option & MACH_SEND_INTERRUPT))
    while (ret == MACH_SEND_INTERRUPTED)
      ret = __mach_msg_trap (msg, option, send_size,
			     rcv_size, rcv_name, timeout, notify);

  if (!(option & MACH_RCV_INTERRUPT))
    while (ret == MACH_RCV_INTERRUPTED)
      ret = __mach_msg_trap (msg, option & ~MACH_SEND_MSG,
			     0, rcv_size, rcv_name, timeout, notify);

  return ret;
}
weak_alias (__mach_msg, mach_msg)
libc_hidden_def (__mach_msg)

mach_msg_return_t
__mach_msg_send	(mach_msg_header_t *msg)
{
  return __mach_msg (msg, MACH_SEND_MSG,
		     msg->msgh_size, 0, MACH_PORT_NULL,
		     MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
}
weak_alias (__mach_msg_send, mach_msg_send)

mach_msg_return_t
__mach_msg_receive (mach_msg_header_t *msg)
{
  return __mach_msg (msg, MACH_RCV_MSG,
		     0, msg->msgh_size, msg->msgh_local_port,
		     MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
}
weak_alias (__mach_msg_receive, mach_msg_receive)
