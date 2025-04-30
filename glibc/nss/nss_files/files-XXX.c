/* Common code for file-based databases in nss_files module.
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

#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libc-lock.h>
#include "nsswitch.h"
#include <nss_files.h>

#include <kernel-features.h>

/* These symbols are defined by the including source file:

   ENTNAME -- database name of the structure and functions (hostent, pwent).
   STRUCTURE -- struct name, define only if not ENTNAME (passwd, group).
   DATABASE -- string of the database file's name ("hosts", "passwd").

   NEED_H_ERRNO - defined iff an arg `int *herrnop' is used.

   Also see files-parse.c.
*/

#define ENTNAME_r	CONCAT(ENTNAME,_r)

#define DATAFILE	"/etc/" DATABASE

#ifdef NEED_H_ERRNO
# include <netdb.h>
# define H_ERRNO_PROTO	, int *herrnop
# define H_ERRNO_ARG	, herrnop
# define H_ERRNO_ARG_OR_NULL herrnop
# define H_ERRNO_SET(val) (*herrnop = (val))
#else
# define H_ERRNO_PROTO
# define H_ERRNO_ARG
# define H_ERRNO_ARG_OR_NULL NULL
# define H_ERRNO_SET(val) ((void) 0)
#endif

#ifndef EXTRA_ARGS
# define EXTRA_ARGS
# define EXTRA_ARGS_DECL
# define EXTRA_ARGS_VALUE
#endif


/* Maintenance of the stream open on the database file.  For getXXent
   operations the stream needs to be held open across calls, the other
   getXXbyYY operations all use their own stream.  */

/* Open database file if not already opened.  */
static enum nss_status
internal_setent (FILE **stream)
{
  enum nss_status status = NSS_STATUS_SUCCESS;

  if (*stream == NULL)
    {
      *stream = __nss_files_fopen (DATAFILE);

      if (*stream == NULL)
	status = errno == EAGAIN ? NSS_STATUS_TRYAGAIN : NSS_STATUS_UNAVAIL;
    }
  else
    rewind (*stream);

  return status;
}


/* Thread-safe, exported version of that.  */
enum nss_status
CONCAT(_nss_files_set,ENTNAME) (int stayopen)
{
  return __nss_files_data_setent (CONCAT (nss_file_, ENTNAME), DATAFILE);
}
libc_hidden_def (CONCAT (_nss_files_set,ENTNAME))

enum nss_status
CONCAT(_nss_files_end,ENTNAME) (void)
{
  return __nss_files_data_endent (CONCAT (nss_file_, ENTNAME));
}
libc_hidden_def (CONCAT (_nss_files_end,ENTNAME))


/* Parsing the database file into `struct STRUCTURE' data structures.  */
static enum nss_status
internal_getent (FILE *stream, struct STRUCTURE *result,
		 char *buffer, size_t buflen, int *errnop H_ERRNO_PROTO
		 EXTRA_ARGS_DECL)
{
  struct parser_data *data = (void *) buffer;
  size_t linebuflen = buffer + buflen - data->linebuffer;
  int saved_errno = errno;	/* Do not clobber errno on success.  */

  if (buflen < sizeof *data + 2)
    {
      *errnop = ERANGE;
      H_ERRNO_SET (NETDB_INTERNAL);
      return NSS_STATUS_TRYAGAIN;
    }

  while (true)
    {
      off64_t original_offset;
      int ret = __nss_readline (stream, data->linebuffer, linebuflen,
				&original_offset);
      if (ret == ENOENT)
	{
	  /* End of file.  */
	  H_ERRNO_SET (HOST_NOT_FOUND);
	  __set_errno (saved_errno);
	  return NSS_STATUS_NOTFOUND;
	}
      else if (ret == 0)
	{
	  ret = __nss_parse_line_result (stream, original_offset,
					 parse_line (data->linebuffer,
						     result, data, buflen,
						     errnop EXTRA_ARGS));
	  if (ret == 0)
	    {
	      /* Line has been parsed successfully.  */
	      __set_errno (saved_errno);
	      return NSS_STATUS_SUCCESS;
	    }
	  else if (ret == EINVAL)
	    /* If it is invalid, loop to get the next line of the file
	       to parse.  */
	    continue;
	}

      *errnop = ret;
      H_ERRNO_SET (NETDB_INTERNAL);
      if (ret == ERANGE)
	/* Request larger buffer.  */
	return NSS_STATUS_TRYAGAIN;
      else
	/* Other read failure.  */
	return NSS_STATUS_UNAVAIL;
    }
}


/* Return the next entry from the database file, doing locking.  */
enum nss_status
CONCAT(_nss_files_get,ENTNAME_r) (struct STRUCTURE *result, char *buffer,
				  size_t buflen, int *errnop H_ERRNO_PROTO)
{
  /* Return next entry in host file.  */

  struct nss_files_per_file_data *data;
  enum nss_status status = __nss_files_data_open (&data,
						  CONCAT (nss_file_, ENTNAME),
						  DATAFILE,
						  errnop, H_ERRNO_ARG_OR_NULL);
  if (status != NSS_STATUS_SUCCESS)
    return status;

  status = internal_getent (data->stream, result, buffer, buflen, errnop
			    H_ERRNO_ARG EXTRA_ARGS_VALUE);

  __nss_files_data_put (data);
  return status;
}
libc_hidden_def (CONCAT (_nss_files_get,ENTNAME_r))

/* Macro for defining lookup functions for this file-based database.

   NAME is the name of the lookup; e.g. `hostbyname'.

   DB_CHAR, KEYPATTERN, KEYSIZE are ignored here but used by db-XXX.c
   e.g. `1 + sizeof (id) * 4'.

   PROTO is the potentially empty list of other parameters.

   BREAK_IF_MATCH is a block of code which compares `struct STRUCTURE *result'
   to the lookup key arguments and does `break;' if they match.  */

#define DB_LOOKUP(name, db_char, keysize, keypattern, break_if_match, proto...)\
enum nss_status								      \
_nss_files_get##name##_r (proto,					      \
			  struct STRUCTURE *result, char *buffer,	      \
			  size_t buflen, int *errnop H_ERRNO_PROTO)	      \
{									      \
  enum nss_status status;						      \
  FILE *stream = NULL;							      \
									      \
  /* Open file.  */							      \
  status = internal_setent (&stream);					      \
									      \
  if (status == NSS_STATUS_SUCCESS)					      \
    {									      \
      while ((status = internal_getent (stream, result, buffer, buflen, errnop \
					H_ERRNO_ARG EXTRA_ARGS_VALUE))	      \
	     == NSS_STATUS_SUCCESS)					      \
	{ break_if_match }						      \
									      \
      fclose (stream);							      \
    }									      \
									      \
  return status;							      \
}									      \
libc_hidden_def (_nss_files_get##name##_r)
