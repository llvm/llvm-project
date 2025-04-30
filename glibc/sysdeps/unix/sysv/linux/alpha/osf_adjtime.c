/* adjtime -- adjust the system clock.  Linux/Alpha/tv32 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#include <time.h>
#include <sys/time.h>
#include <sys/timex.h>
#include <string.h>

struct timex32 {
	unsigned int modes;	/* mode selector */
	long offset;		/* time offset (usec) */
	long freq;		/* frequency offset (scaled ppm) */
	long maxerror;		/* maximum error (usec) */
	long esterror;		/* estimated error (usec) */
	int status;		/* clock command/status */
	long constant;		/* pll time constant */
	long precision;		/* clock precision (usec) (read only) */
	long tolerance;		/* clock frequency tolerance (ppm)
				 * (read only)
				 */
	struct __timeval32 time;	/* (read only) */
	long tick;		/* (modified) usecs between clock ticks */

	long ppsfreq;           /* pps frequency (scaled ppm) (ro) */
	long jitter;            /* pps jitter (us) (ro) */
	int shift;              /* interval duration (s) (shift) (ro) */
	long stabil;            /* pps stability (scaled ppm) (ro) */
	long jitcnt;            /* jitter limit exceeded (ro) */
	long calcnt;            /* calibration intervals (ro) */
	long errcnt;            /* calibration errors (ro) */
	long stbcnt;            /* stability limit exceeded (ro) */

	int  :32; int  :32; int  :32; int  :32;
	int  :32; int  :32; int  :32; int  :32;
	int  :32; int  :32; int  :32; int  :32;
};

int
attribute_compat_text_section
__adjtime_tv32 (const struct __timeval32 *itv, struct __timeval32 *otv)
{
  struct timeval itv64 = valid_timeval32_to_timeval (*itv);
  struct timeval otv64;

  if (__adjtime (&itv64, &otv64) == -1)
    return -1;

  *otv = valid_timeval_to_timeval32 (otv64);
  return 0;
}

int
attribute_compat_text_section
__adjtimex_tv32 (struct timex32 *tx)
{
  struct timex tx64;
  memset (&tx64, 0, sizeof tx64);
  tx64.modes     = tx->modes;
  tx64.offset    = tx->offset;
  tx64.freq      = tx->freq;
  tx64.maxerror  = tx->maxerror;
  tx64.esterror  = tx->esterror;
  tx64.status    = tx->status;
  tx64.constant  = tx->constant;
  tx64.precision = tx->precision;
  tx64.tolerance = tx->tolerance;
  tx64.tick      = tx->tick;
  tx64.ppsfreq   = tx->ppsfreq;
  tx64.jitter    = tx->jitter;
  tx64.shift     = tx->shift;
  tx64.stabil    = tx->stabil;
  tx64.jitcnt    = tx->jitcnt;
  tx64.calcnt    = tx->calcnt;
  tx64.errcnt    = tx->errcnt;
  tx64.stbcnt    = tx->stbcnt;
  tx64.time      = valid_timeval32_to_timeval (tx->time);

  int status = __adjtimex (&tx64);
  if (status < 0)
    return status;

  memset (tx, 0, sizeof *tx);
  tx->modes     = tx64.modes;
  tx->offset    = tx64.offset;
  tx->freq      = tx64.freq;
  tx->maxerror  = tx64.maxerror;
  tx->esterror  = tx64.esterror;
  tx->status    = tx64.status;
  tx->constant  = tx64.constant;
  tx->precision = tx64.precision;
  tx->tolerance = tx64.tolerance;
  tx->tick      = tx64.tick;
  tx->ppsfreq   = tx64.ppsfreq;
  tx->jitter    = tx64.jitter;
  tx->shift     = tx64.shift;
  tx->stabil    = tx64.stabil;
  tx->jitcnt    = tx64.jitcnt;
  tx->calcnt    = tx64.calcnt;
  tx->errcnt    = tx64.errcnt;
  tx->stbcnt    = tx64.stbcnt;
  tx->time      = valid_timeval_to_timeval32 (tx64.time);

  return status;
}

strong_alias (__adjtimex_tv32, __adjtimex_tv32_1);
strong_alias (__adjtimex_tv32, __adjtimex_tv32_2);
compat_symbol (libc, __adjtimex_tv32_1, __adjtimex, GLIBC_2_0);
compat_symbol (libc, __adjtimex_tv32_2, adjtimex, GLIBC_2_0);
compat_symbol (libc, __adjtime_tv32, adjtime, GLIBC_2_0);

#endif /* SHLIB_COMPAT */
