/* Network-related functions for internal library use.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef _NET_INTERNAL_H
#define _NET_INTERNAL_H 1

#include <arpa/inet.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <libc-diag.h>
#include <struct___timespec64.h>

int __inet6_scopeid_pton (const struct in6_addr *address,
                          const char *scope, uint32_t *result);
libc_hidden_proto (__inet6_scopeid_pton)


/* IDNA conversion.  These functions convert domain names between the
   current multi-byte character set and the IDNA encoding.  On
   success, the result string is written to *RESULT (which the caller
   has to free), and zero is returned.  On error, an EAI_* error code
   is returned (see <netdb.h>), and *RESULT is not changed.  */
int __idna_to_dns_encoding (const char *name, char **result);
libc_hidden_proto (__idna_to_dns_encoding)
int __idna_from_dns_encoding (const char *name, char **result);
libc_hidden_proto (__idna_from_dns_encoding)


/* Return value of __idna_name_classify below.  */
enum idna_name_classification
{
  idna_name_ascii,          /* No non-ASCII characters.  */
  idna_name_nonascii,       /* Non-ASCII characters, no backslash.  */
  idna_name_nonascii_backslash, /* Non-ASCII characters with backslash.  */
  idna_name_encoding_error, /* Decoding error.  */
  idna_name_memory_error,   /* Memory allocation failure.  */
  idna_name_error,          /* Other error during decoding.  Check errno.  */
};

/* Check the specified name for non-ASCII characters and backslashes
   or encoding errors.  */
enum idna_name_classification __idna_name_classify (const char *name)
  attribute_hidden;

/* Deadline handling for enforcing timeouts.

   Code should call __deadline_current_time to obtain the current time
   and cache it locally.  The cache needs updating after every
   long-running or potentially blocking operation.  Deadlines relative
   to the current time can be computed using __deadline_from_timeval.
   The deadlines may have to be recomputed in response to certain
   events (such as an incoming packet), but they are absolute (not
   relative to the current time).  A timeout suitable for use with the
   poll function can be computed from such a deadline using
   __deadline_to_ms.

   The fields in the structs defined belowed should only be used
   within the implementation.  */

/* Cache of the current time.  Used to compute deadlines from relative
   timeouts and vice versa.  */
struct deadline_current_time
{
  struct __timespec64 current;
};

/* Return the current time.  Terminates the process if the current
   time is not available.  */
struct deadline_current_time __deadline_current_time (void) attribute_hidden;

/* Computed absolute deadline.  */
struct deadline
{
  struct __timespec64 absolute;
};


/* For internal use only.  */
static inline bool
__deadline_is_infinite (struct deadline deadline)
{
  return deadline.absolute.tv_nsec < 0;
}

/* GCC 8.3 and 9.2 both incorrectly report total_deadline
 * (from sunrpc/clnt_udp.c) as maybe-uninitialized when tv_sec is 8 bytes
 * (64-bits) wide on 32-bit systems. We have to set -Wmaybe-uninitialized
 * here as it won't fix the error in sunrpc/clnt_udp.c.
 * A GCC bug has been filed here:
 *    https://gcc.gnu.org/bugzilla/show_bug.cgi?id=91691
 */
DIAG_PUSH_NEEDS_COMMENT;
DIAG_IGNORE_NEEDS_COMMENT (9, "-Wmaybe-uninitialized");

/* Return true if the current time is at the deadline or past it.  */
static inline bool
__deadline_elapsed (struct deadline_current_time current,
                    struct deadline deadline)
{
  return !__deadline_is_infinite (deadline)
    && (current.current.tv_sec > deadline.absolute.tv_sec
        || (current.current.tv_sec == deadline.absolute.tv_sec
            && current.current.tv_nsec >= deadline.absolute.tv_nsec));
}

/* Return the deadline which occurs first.  */
static inline struct deadline
__deadline_first (struct deadline left, struct deadline right)
{
  if (__deadline_is_infinite (right)
      || left.absolute.tv_sec < right.absolute.tv_sec
      || (left.absolute.tv_sec == right.absolute.tv_sec
          && left.absolute.tv_nsec < right.absolute.tv_nsec))
    return left;
  else
    return right;
}

DIAG_POP_NEEDS_COMMENT;

/* Add TV to the current time and return it.  Returns a special
   infinite absolute deadline on overflow.  */
struct deadline __deadline_from_timeval (struct deadline_current_time,
                                         struct timeval tv) attribute_hidden;

/* Compute the number of milliseconds until the specified deadline,
   from the current time in the argument.  The result is mainly for
   use with poll.  If the deadline has already passed, return 0.  If
   the result would overflow an int, return INT_MAX.  */
int __deadline_to_ms (struct deadline_current_time, struct deadline)
  attribute_hidden;

/* Return true if TV.tv_sec is non-negative and TV.tv_usec is in the
   interval [0, 999999].  */
static inline bool
__is_timeval_valid_timeout (struct timeval tv)
{
  return tv.tv_sec >= 0 && tv.tv_usec >= 0 && tv.tv_usec < 1000 * 1000;
}

#endif /* _NET_INTERNAL_H */
