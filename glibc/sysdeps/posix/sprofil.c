/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.
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
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sigsetops.h>

#include <sys/time.h>
#include <sys/profil.h>

#ifndef SIGPROF
# include <gmon/sprofil.c>
#else

#include <libc-internal.h>

struct region
  {
    size_t offset;
    size_t nsamples;
    unsigned int scale;
    union
      {
	void *vp;
	unsigned short *us;
	unsigned int *ui;
      }
    sample;
    size_t start;
    size_t end;
  };

struct prof_info
  {
    unsigned int num_regions;
    struct region *region;
    struct region *last, *overflow;
    struct itimerval saved_timer;
    struct sigaction saved_action;
  };

static unsigned int overflow_counter;

static struct region default_overflow_region =
  {
    .offset	= 0,
    .nsamples	= 1,
    .scale	= 2,
    .sample	= { &overflow_counter },
    .start	= 0,
    .end	= ~(size_t) 0
  };

static struct prof_info prof_info;

static unsigned long int
pc_to_index (size_t pc, size_t offset, unsigned int scale, int prof_uint)
{
  size_t i = (pc - offset) / (prof_uint ? sizeof (int) : sizeof (short));

  if (sizeof (unsigned long long int) > sizeof (size_t))
    return (unsigned long long int) i * scale / 65536;
  else
    return i / 65536 * scale + i % 65536 * scale / 65536;
}

static inline size_t
index_to_pc (unsigned long int n, size_t offset, unsigned int scale,
	     int prof_uint)
{
  size_t pc, bin_size = (prof_uint ? sizeof (int) : sizeof (short));

  if (sizeof (unsigned long long int) > sizeof (size_t))
    pc = offset + (unsigned long long int) n * bin_size * 65536ull / scale;
  else
    pc = (offset + n * bin_size / scale * 65536
	  + n * bin_size % scale * 65536 / scale);

  if (pc_to_index (pc, offset, scale, prof_uint) < n)
    /* Adjust for rounding error.  */
    ++pc;

  assert (pc_to_index (pc - 1, offset, scale, prof_uint) < n
	  && pc_to_index (pc, offset, scale, prof_uint) >= n);

  return pc;
}

static void
profil_count (uintptr_t pcp, int prof_uint)
{
  struct region *region, *r = prof_info.last;
  size_t lo, hi, mid, pc = pcp;
  unsigned long int i;

  /* Fast path: pc is in same region as before.  */
  if (pc >= r->start && pc < r->end)
    region = r;
  else
    {
      /* Slow path: do a binary search for the right region.  */
      lo = 0; hi = prof_info.num_regions - 1;
      while (lo <= hi)
	{
	  mid = (lo + hi) / 2;

	  r = prof_info.region + mid;
	  if (pc >= r->start && pc < r->end)
	    {
	      prof_info.last = r;
	      region = r;
	      break;
	    }

	  if (pc < r->start)
	    hi = mid - 1;
	  else
	    lo = mid + 1;
	}

      /* No matching region: increment overflow count.  There is no point
	 in updating the cache here, as it won't hit anyhow.  */
      region = prof_info.overflow;
    }

  i = pc_to_index (pc, region->offset, region->scale, prof_uint);
  if (i < r->nsamples)
    {
      if (prof_uint)
	{
	  if (r->sample.ui[i] < (unsigned int) ~0)
	    ++r->sample.ui[i];
	}
      else
	{
	  if (r->sample.us[i] < (unsigned short) ~0)
	    ++r->sample.us[i];
	}
    }
  else
    {
      if (prof_uint)
	++prof_info.overflow->sample.ui[0];
      else
	++prof_info.overflow->sample.us[0];
    }
}

static inline void
profil_count_ushort (uintptr_t pcp)
{
  profil_count (pcp, 0);
}

static inline void
profil_count_uint (uintptr_t pcp)
{
  profil_count (pcp, 1);
}

/* Get the machine-dependent definition of `__profil_counter', the signal
   handler for SIGPROF.  It calls `profil_count' (above) with the PC of the
   interrupted code.  */
#define __profil_counter	__profil_counter_ushort
#define profil_count(pc)	profil_count (pc, 0)
#include <profil-counter.h>

#undef __profil_counter
#undef profil_count

#define __profil_counter	__profil_counter_uint
#define profil_count(pc)	profil_count (pc, 1)
#include <profil-counter.h>

static int
insert (int i, unsigned long int start, unsigned long int end, struct prof *p,
	int prof_uint)
{
  struct region *r;
  size_t to_copy;

  if (start >= end)
    return 0;		/* don't bother with empty regions */

  if (prof_info.num_regions == 0)
    r = malloc (sizeof (*r));
  else
    r = realloc (prof_info.region, (prof_info.num_regions + 1) * sizeof (*r));
  if (r == NULL)
    return -1;

  to_copy = prof_info.num_regions - i;
  if (to_copy > 0)
    memmove (r + i + 1, r + i, to_copy * sizeof (*r));

  r[i].offset = p->pr_off;
  r[i].nsamples = p->pr_size / (prof_uint ? sizeof (int) : sizeof (short));
  r[i].scale = p->pr_scale;
  r[i].sample.vp = p->pr_base;
  r[i].start = start;
  r[i].end = end;

  prof_info.region = r;
  ++prof_info.num_regions;

  if (p->pr_off == 0 && p->pr_scale == 2)
    prof_info.overflow = r;

  return 0;
}

/* Add a new profiling region.  If the new region overlaps with
   existing ones, this may add multiple subregions so that the final
   data structure is free of overlaps.  The absence of overlaps makes
   it possible to use a binary search in profil_count().  Note that
   this function depends on new regions being presented in DECREASING
   ORDER of starting address.  */

static int
add_region (struct prof *p, int prof_uint)
{
  unsigned long int nsamples;
  size_t start, end;
  unsigned int i;

  if (p->pr_scale < 2)
    return 0;

  nsamples = p->pr_size / (prof_uint ? sizeof (int) : sizeof (short));

  start = p->pr_off;
  end = index_to_pc (nsamples, p->pr_off, p->pr_scale, prof_uint);

  /* Merge with existing regions.  */
  for (i = 0; i < prof_info.num_regions; ++i)
    {
      if (start < prof_info.region[i].start)
	{
	  if (end < prof_info.region[i].start)
	    break;
	  else if (insert (i, start, prof_info.region[i].start, p, prof_uint)
		   < 0)
	    return -1;
	}
      start = prof_info.region[i].end;
    }
  return insert (i, start, end, p, prof_uint);
}

static int
pcmp (const void *left, const void *right)
{
  struct prof *l = *(struct prof **) left;
  struct prof *r = *(struct prof **) right;

  if (l->pr_off < r->pr_off)
    return 1;
  else if (l->pr_off > r->pr_off)
    return -1;
  return 0;
}

int
__sprofil (struct prof *profp, int profcnt, struct timeval *tvp,
	   unsigned int flags)
{
  struct prof *p[profcnt];
  struct itimerval timer;
  struct sigaction act;
  int i;

  if (tvp != NULL)
    {
      /* Return profiling period.  */
      unsigned long int t = 1000000 / __profile_frequency ();
      tvp->tv_sec  = t / 1000000;
      tvp->tv_usec = t % 1000000;
    }

  if (prof_info.num_regions > 0)
    {
      /* Disable profiling.  */
      if (__setitimer (ITIMER_PROF, &prof_info.saved_timer, NULL) < 0)
	return -1;

      if (__sigaction (SIGPROF, &prof_info.saved_action, NULL) < 0)
	return -1;

      free (prof_info.region);
      return 0;
    }

  prof_info.num_regions = 0;
  prof_info.region = NULL;
  prof_info.overflow = &default_overflow_region;

  for (i = 0; i < profcnt; ++i)
    p[i] = profp + i;

  /* Sort in order of decreasing starting address: */
  qsort (p, profcnt, sizeof (p[0]), pcmp);

  /* Add regions in order of decreasing starting address: */
  for (i = 0; i < profcnt; ++i)
    if (add_region (p[i], (flags & PROF_UINT) != 0) < 0)
      {
	free (prof_info.region);
	prof_info.num_regions = 0;
	prof_info.region = NULL;
	return -1;
      }

  if (prof_info.num_regions == 0)
    return 0;

  prof_info.last = prof_info.region;

  /* Install SIGPROF handler.  */
#ifdef SA_SIGINFO
  act.sa_sigaction= flags & PROF_UINT
		    ? __profil_counter_uint
		    : __profil_counter_ushort;
  act.sa_flags = SA_SIGINFO;
#else
  act.sa_handler = flags & PROF_UINT
		   ? (sighandler_t) __profil_counter_uint
		   : (sighandler_t) __profil_counter_ushort;
  act.sa_flags = 0;
#endif
  act.sa_flags |= SA_RESTART;
  __sigfillset (&act.sa_mask);
  if (__sigaction (SIGPROF, &act, &prof_info.saved_action) < 0)
    return -1;

  /* Setup profiling timer.  */
  timer.it_value.tv_sec  = 0;
  timer.it_value.tv_usec = 1;
  timer.it_interval = timer.it_value;
  return __setitimer (ITIMER_PROF, &timer, &prof_info.saved_timer);
}

weak_alias (__sprofil, sprofil)

#endif /* SIGPROF */
