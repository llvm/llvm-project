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

#include <assert.h>
#include <errno.h>
#include <libc-lock.h>
#include <stdlib.h>
#include <resolv.h>

#include "nsswitch.h"

/*******************************************************************\
|* Here we assume several symbols to be defined:		   *|
|*								   *|
|* LOOKUP_TYPE   - the return type of the function		   *|
|*								   *|
|* FUNCTION_NAME - name of the non-reentrant function		   *|
|*								   *|
|* DATABASE_NAME - name of the database the function accesses	   *|
|*		   (e.g., host, services, ...)			   *|
|*								   *|
|* ADD_PARAMS    - additional parameter, can vary in number	   *|
|*								   *|
|* ADD_VARIABLES - names of additional parameter		   *|
|*								   *|
|* BUFLEN	 - length of buffer allocated for the non	   *|
|*		   reentrant version				   *|
|*								   *|
|* Optionally the following vars can be defined:		   *|
|*								   *|
|* NEED_H_ERRNO  - an extra parameter will be passed to point to   *|
|*		   the global `h_errno' variable.		   *|
|*								   *|
\*******************************************************************/


#ifdef HANDLE_DIGITS_DOTS
# include <resolv/resolv_context.h>
#endif

/* To make the real sources a bit prettier.  */
#define REENTRANT_NAME APPEND_R (FUNCTION_NAME)
#define APPEND_R(name) APPEND_R1 (name)
#define APPEND_R1(name) name##_r
#define INTERNAL(name) INTERNAL1 (name)
#define INTERNAL1(name) __##name

/* Sometimes we need to store error codes in the `h_errno' variable.  */
#ifdef NEED_H_ERRNO
# define H_ERRNO_PARM , int *h_errnop
# define H_ERRNO_VAR , &h_errno_tmp
# define H_ERRNO_VAR_P &h_errno_tmp
#else
# define H_ERRNO_PARM
# define H_ERRNO_VAR
# define H_ERRNO_VAR_P NULL
#endif

#ifdef HAVE_AF
# define AF_VAL af
#else
# define AF_VAL AF_INET
#endif

/* Prototype for reentrant version we use here.  */
extern int INTERNAL (REENTRANT_NAME) (ADD_PARAMS, LOOKUP_TYPE *resbuf,
				      char *buffer, size_t buflen,
				      LOOKUP_TYPE **result H_ERRNO_PARM)
     attribute_hidden;

/* We need to protect the dynamic buffer handling.  */
__libc_lock_define_initialized (static, lock);

/* This points to the static buffer used.  */
libc_freeres_ptr (static char *buffer);


LOOKUP_TYPE *
FUNCTION_NAME (ADD_PARAMS)
{
  static size_t buffer_size;
  static LOOKUP_TYPE resbuf;
  LOOKUP_TYPE *result;
#ifdef NEED_H_ERRNO
  int h_errno_tmp = 0;
#endif

#ifdef HANDLE_DIGITS_DOTS
  /* Wrap both __nss_hostname_digits_dots and the actual lookup
     function call in the same context.  */
  struct resolv_context *res_ctx = __resolv_context_get ();
  if (res_ctx == NULL)
    {
# if NEED_H_ERRNO
      __set_h_errno (NETDB_INTERNAL);
# endif
      return NULL;
    }
#endif

  /* Get lock.  */
  __libc_lock_lock (lock);

  if (buffer == NULL)
    {
      buffer_size = BUFLEN;
      buffer = (char *) malloc (buffer_size);
    }

#ifdef HANDLE_DIGITS_DOTS
  if (buffer != NULL)
    {
      if (__nss_hostname_digits_dots_context
	  (res_ctx, name, &resbuf, &buffer, &buffer_size, 0, &result, NULL,
	   AF_VAL, H_ERRNO_VAR_P))
	goto done;
    }
#endif

  while (buffer != NULL
	 && (INTERNAL (REENTRANT_NAME) (ADD_VARIABLES, &resbuf, buffer,
					buffer_size, &result H_ERRNO_VAR)
	     == ERANGE)
#ifdef NEED_H_ERRNO
	 && h_errno_tmp == NETDB_INTERNAL
#endif
	 )
    {
      char *new_buf;
      buffer_size *= 2;
      new_buf = (char *) realloc (buffer, buffer_size);
      if (new_buf == NULL)
	{
	  /* We are out of memory.  Free the current buffer so that the
	     process gets a chance for a normal termination.  */
	  free (buffer);
	  __set_errno (ENOMEM);
	}
      buffer = new_buf;
    }

  if (buffer == NULL)
    result = NULL;

#ifdef HANDLE_DIGITS_DOTS
done:
#endif
  /* Release lock.  */
  __libc_lock_unlock (lock);

#ifdef HANDLE_DIGITS_DOTS
  __resolv_context_put (res_ctx);
#endif

#ifdef NEED_H_ERRNO
  if (h_errno_tmp != 0)
    __set_h_errno (h_errno_tmp);
#endif

  return result;
}

nss_interface_function (FUNCTION_NAME)
