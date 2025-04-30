/* Low-level statistical profiling support function.  Mostly POSIX.1 version.
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

#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/time.h>
#include <stdint.h>
#include <libc-internal.h>
#include <sigsetops.h>

#ifndef SIGPROF

#include <gmon/profil.c>

#else

static u_short *samples;
static size_t nsamples;
static size_t pc_offset;
static u_int pc_scale;

static inline void
profil_count (uintptr_t pc)
{
  size_t i = (pc - pc_offset) / 2;

  if (sizeof (unsigned long long int) > sizeof (size_t))
    i = (unsigned long long int) i * pc_scale / 65536;
  else
    i = i / 65536 * pc_scale + i % 65536 * pc_scale / 65536;

  if (i < nsamples)
    ++samples[i];
}

/* Get the machine-dependent definition of `__profil_counter', the signal
   handler for SIGPROF.  It calls `profil_count' (above) with the PC of the
   interrupted code.  */
#include "profil-counter.h"

/* Enable statistical profiling, writing samples of the PC into at most
   SIZE bytes of SAMPLE_BUFFER; every processor clock tick while profiling
   is enabled, the system examines the user PC and increments
   SAMPLE_BUFFER[((PC - OFFSET) / 2) * SCALE / 65536].  If SCALE is zero,
   disable profiling.  Returns zero on success, -1 on error.  */

int
__profil (u_short *sample_buffer, size_t size, size_t offset, u_int scale)
{
  struct sigaction act;
  struct itimerval timer;
#if !IS_IN (rtld)
  static struct sigaction oact;
  static struct itimerval otimer;
# define oact_ptr &oact
# define otimer_ptr &otimer

  if (sample_buffer == NULL)
    {
      /* Disable profiling.  */
      if (samples == NULL)
	/* Wasn't turned on.  */
	return 0;

      if (__setitimer (ITIMER_PROF, &otimer, NULL) < 0)
	return -1;
      samples = NULL;
      return __sigaction (SIGPROF, &oact, NULL);
    }

 if (samples)
    {
      /* Was already turned on.  Restore old timer and signal handler
	 first.  */
      if (__setitimer (ITIMER_PROF, &otimer, NULL) < 0
	  || __sigaction (SIGPROF, &oact, NULL) < 0)
	return -1;
    }
#else
 /* In ld.so profiling should never be disabled once it runs.  */
 //assert (sample_buffer != NULL);
# define oact_ptr NULL
# define otimer_ptr NULL
#endif

  samples = sample_buffer;
  nsamples = size / sizeof *samples;
  pc_offset = offset;
  pc_scale = scale;

#ifdef SA_SIGINFO
  act.sa_sigaction = __profil_counter;
  act.sa_flags = SA_SIGINFO;
#else
  act.sa_handler = __profil_counter;
  act.sa_flags = 0;
#endif
  act.sa_flags |= SA_RESTART;
  __sigfillset (&act.sa_mask);
  if (__sigaction (SIGPROF, &act, oact_ptr) < 0)
    return -1;

  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 1000000 / __profile_frequency ();
  timer.it_interval = timer.it_value;
  return __setitimer (ITIMER_PROF, &timer, otimer_ptr);
}
weak_alias (__profil, profil)

#endif
