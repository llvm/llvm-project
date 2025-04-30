/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, August 1995.

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

#include <sys/msg.h>
#include <errno.h>

/* Read a message from the queue associated with the message queue
   descriptor MSQID.  At most MSGSZ bytes of the message are placed
   in the buffer specified by the MSGP parameter.  The MSGTYP parameter
   describes which message is returned in MSGFLG describes the behaviour
   in buffer overflow or queue underflow.  */

ssize_t
msgrcv (int msqid, void *msgp, size_t msgsz, long msgtyp, int msgflg)
{
  __set_errno (ENOSYS);
  return -1;
}

stub_warning (msgrcv)
