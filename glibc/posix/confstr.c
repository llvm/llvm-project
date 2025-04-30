/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stddef.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <confstr.h>
#include "../version.h"

#define NEED_SPEC_ARRAY 0
#include <posix-conf-vars.h>

/* If BUF is not NULL and LEN > 0, fill in at most LEN - 1 bytes
   of BUF with the value corresponding to NAME and zero-terminate BUF.
   Return the number of bytes required to hold NAME's entire value.  */
size_t
__confstr (int name, char *buf, size_t len)
{
  const char *string = "";
  size_t string_len = 1;

  /* Note that this buffer must be large enough for the longest strings
     used below.  */
  char restenvs[4 * sizeof "POSIX_V7_LPBIG_OFFBIG"];

  switch (name)
    {
    case _CS_PATH:
      {
	static const char cs_path[] = CS_PATH;
	string = cs_path;
	string_len = sizeof (cs_path);
      }
      break;

      /* For _CS_V7_WIDTH_RESTRICTED_ENVS, _CS_V6_WIDTH_RESTRICTED_ENVS
	 and _CS_V5_WIDTH_RESTRICTED_ENVS:

	 We have to return a newline-separated list of names of
	 programming environments in which the widths of blksize_t,
	 cc_t, mode_t, nfds_t, pid_t, ptrdiff_t, size_t, speed_t,
	 ssize_t, suseconds_t, tcflag_t, useconds_t, wchar_t, and
	 wint_t types are no greater than the width of type long.

	 Currently this means all environments that the system allows.  */

#define START_ENV_GROUP(VERSION)		\
    case _CS_##VERSION##_WIDTH_RESTRICTED_ENVS:	\
      string_len = 0;

#define END_ENV_GROUP(VERSION)			\
      restenvs[string_len++] = '\0';		\
      string = restenvs;			\
      break;

#define KNOWN_ABSENT_ENVIRONMENT(SC_PREFIX, ENV_PREFIX, SUFFIX)	\
      /* Empty.  */

#define KNOWN_PRESENT_ENV_STRING(STR)		\
      if (string_len > 0)			\
	restenvs[string_len++] = '\n';		\
      memcpy (restenvs + string_len, STR,	\
	      sizeof STR - 1);			\
      string_len += sizeof STR - 1;

#define KNOWN_PRESENT_ENVIRONMENT(SC_PREFIX, ENV_PREFIX, SUFFIX)	\
      KNOWN_PRESENT_ENV_STRING (#ENV_PREFIX "_" #SUFFIX)

#define UNKNOWN_ENVIRONMENT(SC_PREFIX, ENV_PREFIX, SUFFIX)		\
      if (__sysconf (_SC_##SC_PREFIX##_##SUFFIX) > 0)			\
	{								\
	  KNOWN_PRESENT_ENVIRONMENT (SC_PREFIX, ENV_PREFIX, SUFFIX)	\
	}

#include "posix-envs.def"

#undef START_ENV_GROUP
#undef END_ENV_GROUP
#undef KNOWN_ABSENT_ENVIRONMENT
#undef KNOWN_PRESENT_ENV_STRING
#undef KNOWN_PRESENT_ENVIRONMENT
#undef UNKNOWN_ENVIRONMENT

    case _CS_XBS5_ILP32_OFF32_CFLAGS:
    case _CS_POSIX_V6_ILP32_OFF32_CFLAGS:
    case _CS_POSIX_V7_ILP32_OFF32_CFLAGS:
#ifdef __ILP32_OFF32_CFLAGS
# if CONF_IS_DEFINED_UNSET (_POSIX_V7_ILP32_OFF32)
#  error "__ILP32_OFF32_CFLAGS should not be defined"
# elif CONF_IS_UNDEFINED (_POSIX_V7_ILP32_OFF32)
      if (__sysconf (_SC_V7_ILP32_OFF32) < 0)
	break;
# endif
      string = __ILP32_OFF32_CFLAGS;
      string_len = sizeof (__ILP32_OFF32_CFLAGS);
#endif
      break;

    case _CS_XBS5_ILP32_OFFBIG_CFLAGS:
    case _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS:
    case _CS_POSIX_V7_ILP32_OFFBIG_CFLAGS:
#ifdef __ILP32_OFFBIG_CFLAGS
# if CONF_IS_DEFINED_UNSET (_POSIX_V7_ILP32_OFFBIG)
#  error "__ILP32_OFFBIG_CFLAGS should not be defined"
# elif CONF_IS_UNDEFINED (_POSIX_V7_ILP32_OFFBIG)
      if (__sysconf (_SC_V7_ILP32_OFFBIG) < 0)
	break;
# endif
      string = __ILP32_OFFBIG_CFLAGS;
      string_len = sizeof (__ILP32_OFFBIG_CFLAGS);
#endif
      break;

    case _CS_XBS5_LP64_OFF64_CFLAGS:
    case _CS_POSIX_V6_LP64_OFF64_CFLAGS:
    case _CS_POSIX_V7_LP64_OFF64_CFLAGS:
#ifdef __LP64_OFF64_CFLAGS
# if CONF_IS_DEFINED_UNSET (_POSIX_V7_LP64_OFF64)
#  error "__LP64_OFF64_CFLAGS should not be defined"
# elif CONF_IS_UNDEFINED (_POSIX_V7_LP64_OFF64)
      if (__sysconf (_SC_V7_LP64_OFF64) < 0)
	break;
# endif
      string = __LP64_OFF64_CFLAGS;
      string_len = sizeof (__LP64_OFF64_CFLAGS);
#endif
      break;

    case _CS_XBS5_ILP32_OFF32_LDFLAGS:
    case _CS_POSIX_V6_ILP32_OFF32_LDFLAGS:
    case _CS_POSIX_V7_ILP32_OFF32_LDFLAGS:
#ifdef __ILP32_OFF32_LDFLAGS
# if CONF_IS_DEFINED_UNSET (_POSIX_V7_ILP32_OFF32 )
#  error "__ILP32_OFF32_LDFLAGS should not be defined"
# elif CONF_IS_UNDEFINED (_POSIX_V7_ILP32_OFF32)
      if (__sysconf (_SC_V7_ILP32_OFF32) < 0)
	break;
# endif
      string = __ILP32_OFF32_LDFLAGS;
      string_len = sizeof (__ILP32_OFF32_LDFLAGS);
#endif
      break;

    case _CS_XBS5_ILP32_OFFBIG_LDFLAGS:
    case _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS:
    case _CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS:
#ifdef __ILP32_OFFBIG_LDFLAGS
# if CONF_IS_DEFINED_UNSET (_POSIX_V7_ILP32_OFFBIG)
#  error "__ILP32_OFFBIG_LDFLAGS should not be defined"
# elif CONF_IS_UNDEFINED (_POSIX_V7_ILP32_OFFBIG)
      if (__sysconf (_SC_V7_ILP32_OFFBIG) < 0)
	break;
# endif
      string = __ILP32_OFFBIG_LDFLAGS;
      string_len = sizeof (__ILP32_OFFBIG_LDFLAGS);
#endif
      break;

    case _CS_XBS5_LP64_OFF64_LDFLAGS:
    case _CS_POSIX_V6_LP64_OFF64_LDFLAGS:
    case _CS_POSIX_V7_LP64_OFF64_LDFLAGS:
#ifdef __LP64_OFF64_LDFLAGS
# if CONF_IS_DEFINED_UNSET (_POSIX_V7_LP64_OFF64)
#  error "__LP64_OFF64_LDFLAGS should not be defined"
# elif CONF_IS_UNDEFINED (_POSIX_V7_LP64_OFF64)
      if (__sysconf (_SC_V7_LP64_OFF64) < 0)
	break;
# endif
      string = __LP64_OFF64_LDFLAGS;
      string_len = sizeof (__LP64_OFF64_LDFLAGS);
#endif
      break;

    case _CS_LFS_CFLAGS:
    case _CS_LFS_LINTFLAGS:
#if (CONF_IS_DEFINED_SET (_POSIX_V6_ILP32_OFF32) \
     && CONF_IS_DEFINED_SET (_POSIX_V6_ILP32_OFFBIG))
# define __LFS_CFLAGS "-D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64"
      /* Signal that we want the new ABI.  */
      string = __LFS_CFLAGS;
      string_len = sizeof (__LFS_CFLAGS);
#endif
      break;

    case _CS_LFS_LDFLAGS:
    case _CS_LFS_LIBS:
      /* No special libraries or linker flags needed.  */
      break;

    case _CS_LFS64_CFLAGS:
    case _CS_LFS64_LINTFLAGS:
#define __LFS64_CFLAGS "-D_LARGEFILE64_SOURCE"
      string = __LFS64_CFLAGS;
      string_len = sizeof (__LFS64_CFLAGS);
      break;

    case _CS_LFS64_LDFLAGS:
    case _CS_LFS64_LIBS:
      /* No special libraries or linker flags needed.  */
      break;

    case _CS_XBS5_ILP32_OFF32_LIBS:
    case _CS_XBS5_ILP32_OFF32_LINTFLAGS:
    case _CS_XBS5_ILP32_OFFBIG_LIBS:
    case _CS_XBS5_ILP32_OFFBIG_LINTFLAGS:
    case _CS_XBS5_LP64_OFF64_LIBS:
    case _CS_XBS5_LP64_OFF64_LINTFLAGS:
    case _CS_XBS5_LPBIG_OFFBIG_CFLAGS:
    case _CS_XBS5_LPBIG_OFFBIG_LDFLAGS:
    case _CS_XBS5_LPBIG_OFFBIG_LIBS:
    case _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS:

    case _CS_POSIX_V6_ILP32_OFF32_LIBS:
    case _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS:
    case _CS_POSIX_V6_ILP32_OFFBIG_LIBS:
    case _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS:
    case _CS_POSIX_V6_LP64_OFF64_LIBS:
    case _CS_POSIX_V6_LP64_OFF64_LINTFLAGS:
    case _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS:
    case _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS:
    case _CS_POSIX_V6_LPBIG_OFFBIG_LIBS:
    case _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS:

    case _CS_POSIX_V7_ILP32_OFF32_LIBS:
    case _CS_POSIX_V7_ILP32_OFF32_LINTFLAGS:
    case _CS_POSIX_V7_ILP32_OFFBIG_LIBS:
    case _CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS:
    case _CS_POSIX_V7_LP64_OFF64_LIBS:
    case _CS_POSIX_V7_LP64_OFF64_LINTFLAGS:
    case _CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS:
    case _CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS:
    case _CS_POSIX_V7_LPBIG_OFFBIG_LIBS:
    case _CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS:
      /* GNU libc does not require special actions to use LFS functions.  */
      break;

    case _CS_GNU_LIBC_VERSION:
      string = "glibc " VERSION;
      string_len = sizeof ("glibc " VERSION);
      break;

    case _CS_GNU_LIBPTHREAD_VERSION:
#ifdef LIBPTHREAD_VERSION
      string = LIBPTHREAD_VERSION;
      string_len = sizeof LIBPTHREAD_VERSION;
      break;
#else
      /* No thread library.  */
      __set_errno (EINVAL);
      return 0;
#endif

    case _CS_V6_ENV:
    case _CS_V7_ENV:
      /* Maybe something else is needed in future.  */
      string = "POSIXLY_CORRECT=1";
      string_len = sizeof ("POSIXLY_CORRECT=1");
      break;

    default:
      __set_errno (EINVAL);
      return 0;
    }

  if (len > 0 && buf != NULL)
    {
      if (string_len <= len)
	memcpy (buf, string, string_len);
      else
	{
	  memcpy (buf, string, len - 1);
	  buf[len - 1] = '\0';
	}
    }
  return string_len;
}
libc_hidden_def (__confstr)
libc_hidden_def (confstr)
weak_alias (__confstr, confstr)
