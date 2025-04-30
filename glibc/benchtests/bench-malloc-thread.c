/* Benchmark malloc and free functions.
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

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "bench-timing.h"
#include "json-lib.h"

/* Benchmark duration in seconds.  */
#define BENCHMARK_DURATION	10
#define RAND_SEED		88

#ifndef NUM_THREADS
# define NUM_THREADS 1
#endif

/* Maximum memory that can be allocated at any one time is:

   NUM_THREADS * WORKING_SET_SIZE * MAX_ALLOCATION_SIZE

   However due to the distribution of the random block sizes
   the typical amount allocated will be much smaller.  */
#define WORKING_SET_SIZE	1024

#define MIN_ALLOCATION_SIZE	4
#define MAX_ALLOCATION_SIZE	32768

/* Get a random block size with an inverse square distribution.  */
static unsigned int
get_block_size (unsigned int rand_data)
{
  /* Inverse square.  */
  const float exponent = -2;
  /* Minimum value of distribution.  */
  const float dist_min = MIN_ALLOCATION_SIZE;
  /* Maximum value of distribution.  */
  const float dist_max = MAX_ALLOCATION_SIZE;

  float min_pow = powf (dist_min, exponent + 1);
  float max_pow = powf (dist_max, exponent + 1);

  float r = (float) rand_data / RAND_MAX;

  return (unsigned int) powf ((max_pow - min_pow) * r + min_pow,
			      1 / (exponent + 1));
}

#define NUM_BLOCK_SIZES	8000
#define NUM_OFFSETS	((WORKING_SET_SIZE) * 4)

static unsigned int random_block_sizes[NUM_BLOCK_SIZES];
static unsigned int random_offsets[NUM_OFFSETS];

static void
init_random_values (void)
{
  for (size_t i = 0; i < NUM_BLOCK_SIZES; i++)
    random_block_sizes[i] = get_block_size (rand ());

  for (size_t i = 0; i < NUM_OFFSETS; i++)
    random_offsets[i] = rand () % WORKING_SET_SIZE;
}

static unsigned int
get_random_block_size (unsigned int *state)
{
  unsigned int idx = *state;

  if (idx >= NUM_BLOCK_SIZES - 1)
    idx = 0;
  else
    idx++;

  *state = idx;

  return random_block_sizes[idx];
}

static unsigned int
get_random_offset (unsigned int *state)
{
  unsigned int idx = *state;

  if (idx >= NUM_OFFSETS - 1)
    idx = 0;
  else
    idx++;

  *state = idx;

  return random_offsets[idx];
}

static volatile bool timeout;

static void
alarm_handler (int signum)
{
  timeout = true;
}

/* Allocate and free blocks in a random order.  */
static size_t
malloc_benchmark_loop (void **ptr_arr)
{
  unsigned int offset_state = 0, block_state = 0;
  size_t iters = 0;

  while (!timeout)
    {
      unsigned int next_idx = get_random_offset (&offset_state);
      unsigned int next_block = get_random_block_size (&block_state);

      free (ptr_arr[next_idx]);

      ptr_arr[next_idx] = malloc (next_block);

      iters++;
    }

  return iters;
}

struct thread_args
{
  size_t iters;
  void **working_set;
  timing_t elapsed;
};

static void *
benchmark_thread (void *arg)
{
  struct thread_args *args = (struct thread_args *) arg;
  size_t iters;
  void *thread_set = args->working_set;
  timing_t start, stop;

  TIMING_NOW (start);
  iters = malloc_benchmark_loop (thread_set);
  TIMING_NOW (stop);

  TIMING_DIFF (args->elapsed, start, stop);
  args->iters = iters;

  return NULL;
}

static timing_t
do_benchmark (size_t num_threads, size_t *iters)
{
  timing_t elapsed = 0;

  if (num_threads == 1)
    {
      timing_t start, stop;
      void *working_set[WORKING_SET_SIZE];

      memset (working_set, 0, sizeof (working_set));

      TIMING_NOW (start);
      *iters = malloc_benchmark_loop (working_set);
      TIMING_NOW (stop);

      TIMING_DIFF (elapsed, start, stop);
    }
  else
    {
      struct thread_args args[num_threads];
      void *working_set[num_threads][WORKING_SET_SIZE];
      pthread_t threads[num_threads];

      memset (working_set, 0, sizeof (working_set));

      *iters = 0;

      for (size_t i = 0; i < num_threads; i++)
	{
	  args[i].working_set = working_set[i];
	  pthread_create(&threads[i], NULL, benchmark_thread, &args[i]);
	}

      for (size_t i = 0; i < num_threads; i++)
	{
	  pthread_join(threads[i], NULL);
	  TIMING_ACCUM (elapsed, args[i].elapsed);
	  *iters += args[i].iters;
	}
    }
  return elapsed;
}

static void usage(const char *name)
{
  fprintf (stderr, "%s: <num_threads>\n", name);
  exit (1);
}

int
main (int argc, char **argv)
{
  timing_t cur;
  size_t iters = 0, num_threads = 1;
  json_ctx_t json_ctx;
  double d_total_s, d_total_i;
  struct sigaction act;

  if (argc == 1)
    num_threads = 1;
  else if (argc == 2)
    {
      long ret;

      errno = 0;
      ret = strtol(argv[1], NULL, 10);

      if (errno || ret == 0)
	usage(argv[0]);

      num_threads = ret;
    }
  else
    usage(argv[0]);

  init_random_values ();

  json_init (&json_ctx, 0, stdout);

  json_document_begin (&json_ctx);

  json_attr_string (&json_ctx, "timing_type", TIMING_TYPE);

  json_attr_object_begin (&json_ctx, "functions");

  json_attr_object_begin (&json_ctx, "malloc");

  json_attr_object_begin (&json_ctx, "");

  memset (&act, 0, sizeof (act));
  act.sa_handler = &alarm_handler;

  sigaction (SIGALRM, &act, NULL);

  alarm (BENCHMARK_DURATION);

  cur = do_benchmark (num_threads, &iters);

  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);

  d_total_s = cur;
  d_total_i = iters;

  json_attr_double (&json_ctx, "duration", d_total_s);
  json_attr_double (&json_ctx, "iterations", d_total_i);
  json_attr_double (&json_ctx, "time_per_iteration", d_total_s / d_total_i);
  json_attr_double (&json_ctx, "max_rss", usage.ru_maxrss);

  json_attr_double (&json_ctx, "threads", num_threads);
  json_attr_double (&json_ctx, "min_size", MIN_ALLOCATION_SIZE);
  json_attr_double (&json_ctx, "max_size", MAX_ALLOCATION_SIZE);
  json_attr_double (&json_ctx, "random_seed", RAND_SEED);

  json_attr_object_end (&json_ctx);

  json_attr_object_end (&json_ctx);

  json_attr_object_end (&json_ctx);

  json_document_end (&json_ctx);

  return 0;
}
