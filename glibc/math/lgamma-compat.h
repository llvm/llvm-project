/* ABI compatibility for lgamma functions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef LGAMMA_COMPAT_H
#define LGAMMA_COMPAT_H 1

#include <math-svid-compat.h>
#include <shlib-compat.h>

/* XSI POSIX requires lgamma to set signgam, but ISO C does not permit
   this.  Namespace issues can be avoided if the functions set
   __signgam and signgam is a weak alias, but this only works if both
   signgam and __signgam were exported from the glibc version the
   program was linked against.  Before glibc 2.23, lgamma functions
   set signgam which was not a weak alias for __signgam, so old
   binaries have dynamic symbols for signgam only and the versions of
   lgamma used for old binaries must set both signgam and __signgam.
   Those versions also do a check of _LIB_VERSION != _ISOC_ to match
   old glibc.

   Users of this file define USE_AS_COMPAT to 0 when building the main
   version of lgamma, 1 when building the compatibility version.  */

#define LGAMMA_OLD_VER GLIBC_2_0
#define LGAMMA_NEW_VER GLIBC_2_23
#define HAVE_LGAMMA_COMPAT SHLIB_COMPAT (libm, LGAMMA_OLD_VER, LGAMMA_NEW_VER)

/* Whether to build this version at all.  */
#define BUILD_LGAMMA \
  (LIBM_SVID_COMPAT && (HAVE_LGAMMA_COMPAT || !USE_AS_COMPAT))

/* The name to use for this version.  */
#if USE_AS_COMPAT
# define LGFUNC(FUNC) FUNC ## _compat
#else
# define LGFUNC(FUNC) FUNC
#endif

/* If there is a compatibility version, gamma (not an ISO C function,
   so never a problem for it to set signgam) points directly to it
   rather than having separate versions.  */
#define GAMMA_ALIAS (USE_AS_COMPAT ? HAVE_LGAMMA_COMPAT : !HAVE_LGAMMA_COMPAT)

/* How to call the underlying lgamma_r function.  */
#define CALL_LGAMMA(TYPE, FUNC, ARG)			\
  ({							\
    TYPE lgamma_tmp;					\
    int local_signgam;					\
    if (USE_AS_COMPAT)					\
      {							\
	lgamma_tmp = FUNC ((ARG), &local_signgam);	\
	if (_LIB_VERSION != _ISOC_)			\
	  signgam = __signgam = local_signgam;		\
      }							\
    else						\
      lgamma_tmp = FUNC ((ARG), &__signgam);		\
    lgamma_tmp;						\
  })

#endif /* lgamma-compat.h.  */
