/* Benchmark malloc and free functions.
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
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/resource.h>
#include "bench-timing.h"
#include "json-lib.h"

/* Benchmark the malloc/free performance of a varying number of blocks of a
   given size.  This enables performance tracking of the t-cache and fastbins.
   It tests 3 different scenarios: single-threaded using main arena,
   multi-threaded using thread-arena, and main arena with SINGLE_THREAD_P
   false.  */

#define NUM_ITERS 200000
#define NUM_ALLOCS 4
#define MAX_ALLOCS 1600

typedef struct
{
  size_t iters;
  size_t size;
  int n;
  timing_t elapsed;
} malloc_args;

static void
do_benchmark (malloc_args *args, int **arr)
{
  timing_t start, stop;
  size_t iters = args->iters;
  size_t size = args->size;
  int n = args->n;

  TIMING_NOW (start);

  for (int j = 0; j < iters; j++)
    {
      for (int i = 0; i < n; i++)
	arr[i] = malloc (size);

      for (int i = 0; i < n; i++)
	free (arr[i]);
    }

  TIMING_NOW (stop);

  TIMING_DIFF (args->elapsed, start, stop);
}

static malloc_args tests[3][NUM_ALLOCS];
static int allocs[NUM_ALLOCS] = { 25, 100, 400, MAX_ALLOCS };

static void *
thread_test (void *p)
{
  int **arr = (int**)p;

  /* Run benchmark multi-threaded.  */
  for (int i = 0; i < NUM_ALLOCS; i++)
    do_benchmark (&tests[2][i], arr);

  return p;
}

void
bench (unsigned long size)
{
  size_t iters = NUM_ITERS;
  int **arr = (int**) malloc (MAX_ALLOCS * sizeof (void*));

  for (int t = 0; t < 3; t++)
    for (int i = 0; i < NUM_ALLOCS; i++)
      {
	tests[t][i].n = allocs[i];
	tests[t][i].size = size;
	tests[t][i].iters = iters / allocs[i];

	/* Do a quick warmup run.  */
	if (t == 0)
	  do_benchmark (&tests[0][i], arr);
      }

  /* Run benchmark single threaded in main_arena.  */
  for (int i = 0; i < NUM_ALLOCS; i++)
    do_benchmark (&tests[0][i], arr);

  /* Run benchmark in a thread_arena.  */
  pthread_t t;
  pthread_create (&t, NULL, thread_test, (void*)arr);
  pthread_join (t, NULL);

  /* Repeat benchmark in main_arena with SINGLE_THREAD_P == false.  */
  for (int i = 0; i < NUM_ALLOCS; i++)
    do_benchmark (&tests[1][i], arr);

  free (arr);

  json_ctx_t json_ctx;

  json_init (&json_ctx, 0, stdout);

  json_document_begin (&json_ctx);

  json_attr_string (&json_ctx, "timing_type", TIMING_TYPE);

  json_attr_object_begin (&json_ctx, "functions");

  json_attr_object_begin (&json_ctx, "malloc");

  char s[100];
  double iters2 = iters;

  json_attr_object_begin (&json_ctx, "");
  json_attr_double (&json_ctx, "malloc_block_size", size);

  struct rusage usage;
  getrusage (RUSAGE_SELF, &usage);
  json_attr_double (&json_ctx, "max_rss", usage.ru_maxrss);

  for (int i = 0; i < NUM_ALLOCS; i++)
    {
      sprintf (s, "main_arena_st_allocs_%04d_time", allocs[i]);
      json_attr_double (&json_ctx, s, tests[0][i].elapsed / iters2);
    }

  for (int i = 0; i < NUM_ALLOCS; i++)
    {
      sprintf (s, "main_arena_mt_allocs_%04d_time", allocs[i]);
      json_attr_double (&json_ctx, s, tests[1][i].elapsed / iters2);
    }

  for (int i = 0; i < NUM_ALLOCS; i++)
    {
      sprintf (s, "thread_arena__allocs_%04d_time", allocs[i]);
      json_attr_double (&json_ctx, s, tests[2][i].elapsed / iters2);
    }

  json_attr_object_end (&json_ctx);

  json_attr_object_end (&json_ctx);

  json_attr_object_end (&json_ctx);

  json_document_end (&json_ctx);
}

static void usage (const char *name)
{
  fprintf (stderr, "%s: <alloc_size>\n", name);
  exit (1);
}

int
main (int argc, char **argv)
{
  long val = 16;
  if (argc == 2)
    val = strtol (argv[1], NULL, 0);

  if (argc > 2 || val <= 0)
    usage (argv[0]);

  bench (val);

  return 0;
}
