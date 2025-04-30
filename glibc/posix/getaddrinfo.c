/* Stub version of getaddrinfo function.
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

#include <errno.h>
#include <netdb.h>

int
getaddrinfo (const char *name, const char *service, const struct addrinfo *req,
	     struct addrinfo **pai)
{
  __set_errno (ENOSYS);
  return EAI_SYSTEM;
}
stub_warning (getaddrinfo)
libc_hidden_def (getaddrinfo)

void
freeaddrinfo (struct addrinfo *ai)
{
  /* Nothing.  */
}
stub_warning (freeaddrinfo)
libc_hidden_def (freeaddrinfo)
