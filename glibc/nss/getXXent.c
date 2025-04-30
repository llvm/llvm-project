/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <libc-lock.h>
#include <stdlib.h>

#include "nsswitch.h"

/*******************************************************************\
|* Here we assume several symbols to be defined:		   *|
|*								   *|
|* LOOKUP_TYPE   - the return type of the function		   *|
|*								   *|
|* GETFUNC_NAME  - name of the non-reentrant getXXXent function	   *|
|*								   *|
|* BUFLEN	 - size of static buffer			   *|
|*								   *|
|* Optionally the following vars can be defined:		   *|
|*								   *|
|* NEED_H_ERRNO  - an extra parameter will be passed to point to   *|
|*		   the global `h_errno' variable.		   *|
|*								   *|
\*******************************************************************/

/* To make the real sources a bit prettier.  */
#define REENTRANT_GETNAME APPEND_R (GETFUNC_NAME)
#define APPEND_R(name) APPEND_R1 (name)
#define APPEND_R1(name) name##_r
#define INTERNAL(name) INTERNAL1 (name)
#define INTERNAL1(name) __##name

/* Sometimes we need to store error codes in the `h_errno' variable.  */
#ifdef NEED_H_ERRNO
# define H_ERRNO_PARM , int *h_errnop
# define H_ERRNO_VAR &h_errno
#else
# define H_ERRNO_PARM
# define H_ERRNO_VAR NULL
#endif

/* Prototype of the reentrant version.  */
extern int INTERNAL (REENTRANT_GETNAME) (LOOKUP_TYPE *resbuf, char *buffer,
					 size_t buflen, LOOKUP_TYPE **result
					 H_ERRNO_PARM) attribute_hidden;

/* We need to protect the dynamic buffer handling.  */
__libc_lock_define_initialized (static, lock);

/* This points to the static buffer used.  */
libc_freeres_ptr (static char *buffer);


LOOKUP_TYPE *
GETFUNC_NAME (void)
{
  static size_t buffer_size;
  static union
  {
    LOOKUP_TYPE l;
    void *ptr;
  } resbuf;
  LOOKUP_TYPE *result;
  int save;

  /* Get lock.  */
  __libc_lock_lock (lock);

  result = (LOOKUP_TYPE *)
    __nss_getent ((getent_r_function) INTERNAL (REENTRANT_GETNAME),
		  &resbuf.ptr, &buffer, BUFLEN, &buffer_size,
		  H_ERRNO_VAR);

  save = errno;
  __libc_lock_unlock (lock);
  __set_errno (save);
  return result;
}

nss_interface_function (GETFUNC_NAME)
