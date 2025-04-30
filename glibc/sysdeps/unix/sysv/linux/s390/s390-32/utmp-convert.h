/* Copyright (C) 2008-2021 Free Software Foundation, Inc.
   Contributed by Andreas Krebbel <Andreas.Krebbel@de.ibm.com>.
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


/* This file provides functions converting between the 32 and 64 bit
   struct utmp variants.  */

#ifndef _UTMP_CONVERT_H
#define _UTMP_CONVERT_H 1

#include <string.h>

#include "utmp32.h"

/* Convert the 64 bit struct utmp value in FROM to the 32 bit version
   returned in TO.  */
static inline void
utmp_convert64to32 (const struct utmp *from, struct utmp32 *to)
{
#if _HAVE_UT_TYPE - 0
  to->ut_type = from->ut_type;
#endif
#if _HAVE_UT_PID - 0
  to->ut_pid = from->ut_pid;
#endif
  memcpy (to->ut_line, from->ut_line, UT_LINESIZE);
  memcpy (to->ut_user, from->ut_user, UT_NAMESIZE);
#if _HAVE_UT_ID - 0
  memcpy (to->ut_id, from->ut_id, 4);
#endif
#if _HAVE_UT_HOST - 0
  memcpy (to->ut_host, from->ut_host, UT_HOSTSIZE);
#endif
  to->ut_exit = from->ut_exit;
  to->ut_session = (int32_t) from->ut_session;
#if _HAVE_UT_TV - 0
  to->ut_tv.tv_sec = (int32_t) from->ut_tv.tv_sec;
  to->ut_tv.tv_usec = (int32_t) from->ut_tv.tv_usec;
#endif
  memcpy (to->ut_addr_v6, from->ut_addr_v6, 4 * 4);
}

/* Convert the 32 bit struct utmp value in FROM to the 64 bit version
   returned in TO.  */
static inline void
utmp_convert32to64 (const struct utmp32 *from, struct utmp *to)
{
#if _HAVE_UT_TYPE - 0
  to->ut_type = from->ut_type;
#endif
#if _HAVE_UT_PID - 0
  to->ut_pid = from->ut_pid;
#endif
  memcpy (to->ut_line, from->ut_line, UT_LINESIZE);
  memcpy (to->ut_user, from->ut_user, UT_NAMESIZE);
#if _HAVE_UT_ID - 0
  memcpy (to->ut_id, from->ut_id, 4);
#endif
#if _HAVE_UT_HOST - 0
  memcpy (to->ut_host, from->ut_host, UT_HOSTSIZE);
#endif
  to->ut_exit = from->ut_exit;
  to->ut_session = (int64_t) from->ut_session;
#if _HAVE_UT_TV - 0
  to->ut_tv.tv_sec = (int64_t) from->ut_tv.tv_sec;
  to->ut_tv.tv_usec = (int64_t) from->ut_tv.tv_usec;
#endif
  memcpy (to->ut_addr_v6, from->ut_addr_v6, 4 * 4);
}

#endif /* utmp-convert.h */
