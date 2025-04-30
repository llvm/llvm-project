/* Skeleton for benchmark programs.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include "bench-timing.h"
#include "json-lib.h"
#include "bench-util.h"

#include "bench-util.c"

#define TIMESPEC_AFTER(a, b) \
  (((a).tv_sec == (b).tv_sec)						      \
   ? ((a).tv_nsec > (b).tv_nsec)					      \
   : ((a).tv_sec > (b).tv_sec))
int
main (int argc, char **argv)
{
  unsigned long i, k;
  struct timespec runtime;
  timing_t start, end;
  bool detailed = false;
  json_ctx_t json_ctx;

  if (argc == 2 && !strcmp (argv[1], "-d"))
    detailed = true;

  bench_start ();

  memset (&runtime, 0, sizeof (runtime));

  unsigned long iters = 1000;

#ifdef BENCH_INIT
  BENCH_INIT ();
#endif

  json_init (&json_ctx, 2, stdout);

  /* Begin function.  */
  json_attr_object_begin (&json_ctx, FUNCNAME);

  for (int v = 0; v < NUM_VARIANTS; v++)
    {
      /* Run for approximately DURATION seconds.  */
      clock_gettime (CLOCK_MONOTONIC_RAW, &runtime);
      runtime.tv_sec += DURATION;

      bool is_bench = strncmp (VARIANT (v), "workload-", 9) == 0;
      double d_total_i = 0;
      timing_t total = 0, max = 0, min = 0x7fffffffffffffff;
      timing_t throughput = 0, latency = 0;
      int64_t c = 0;
      uint64_t cur;
      BENCH_VARS;
      while (1)
	{
	  if (is_bench)
	    {
	      /* Benchmark a real trace of calls - all samples are iterated
		 over once before repeating.  This models actual use more
		 accurately than repeating the same sample many times.  */
	      TIMING_NOW (start);
	      for (k = 0; k < iters; k++)
		for (i = 0; i < NUM_SAMPLES (v); i++)
		  BENCH_FUNC (v, i);
	      TIMING_NOW (end);
	      TIMING_DIFF (cur, start, end);
	      TIMING_ACCUM (throughput, cur);

	      TIMING_NOW (start);
	      for (k = 0; k < iters; k++)
		for (i = 0; i < NUM_SAMPLES (v); i++)
		  BENCH_FUNC_LAT (v, i);
	      TIMING_NOW (end);
	      TIMING_DIFF (cur, start, end);
	      TIMING_ACCUM (latency, cur);

	      d_total_i += iters * NUM_SAMPLES (v);
	    }
	  else
	    for (i = 0; i < NUM_SAMPLES (v); i++)
	      {
		TIMING_NOW (start);
		for (k = 0; k < iters; k++)
		  BENCH_FUNC (v, i);
		TIMING_NOW (end);

		TIMING_DIFF (cur, start, end);

		if (cur > max)
		  max = cur;

		if (cur < min)
		  min = cur;

		TIMING_ACCUM (total, cur);
		/* Accumulate timings for the value.  In the end we will divide
		   by the total iterations.  */
		RESULT_ACCUM (cur, v, i, c * iters, (c + 1) * iters);

		d_total_i += iters;
	      }
	  c++;
	  struct timespec curtime;

	  memset (&curtime, 0, sizeof (curtime));
	  clock_gettime (CLOCK_MONOTONIC_RAW, &curtime);
	  if (TIMESPEC_AFTER (curtime, runtime))
	    goto done;
	}

      double d_total_s;
      double d_iters;

    done:
      d_total_s = total;
      d_iters = iters;

      /* Begin variant.  */
      json_attr_object_begin (&json_ctx, VARIANT (v));

      if (is_bench)
	{
	  json_attr_double (&json_ctx, "duration", throughput + latency);
	  json_attr_double (&json_ctx, "iterations", 2 * d_total_i);
	  json_attr_double (&json_ctx, "reciprocal-throughput",
			    throughput / d_total_i);
	  json_attr_double (&json_ctx, "latency", latency / d_total_i);
	  json_attr_double (&json_ctx, "max-throughput",
			    d_total_i / throughput * 1000000000.0);
	  json_attr_double (&json_ctx, "min-throughput",
			    d_total_i / latency * 1000000000.0);
	}
      else
	{
	  json_attr_double (&json_ctx, "duration", d_total_s);
	  json_attr_double (&json_ctx, "iterations", d_total_i);
	  json_attr_double (&json_ctx, "max", max / d_iters);
	  json_attr_double (&json_ctx, "min", min / d_iters);
	  json_attr_double (&json_ctx, "mean", d_total_s / d_total_i);
	}

      if (detailed && !is_bench)
	{
	  json_array_begin (&json_ctx, "timings");

	  for (int i = 0; i < NUM_SAMPLES (v); i++)
	    json_element_double (&json_ctx, RESULT (v, i));

	  json_array_end (&json_ctx);
	}

      /* End variant.  */
      json_attr_object_end (&json_ctx);
    }

  /* End function.  */
  json_attr_object_end (&json_ctx);

  return 0;
}
