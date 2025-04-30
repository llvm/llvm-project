/* Test and measure string and memory functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.

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

#include <sys/cdefs.h>
#include <support/support.h>

typedef struct
{
  const char *name;
  void (*fn) (void);
  long test;
} impl_t;
extern impl_t __start_impls[], __stop_impls[];

#define IMPL(name, test) \
  impl_t tst_ ## name							\
  __attribute__ ((section ("impls"), aligned (sizeof (void *))))	\
       = { __STRING (name), (void (*) (void))name, test };

#ifdef TEST_MAIN

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#undef __USE_STRING_INLINES

/* We are compiled under _ISOMAC, so libc-symbols.h does not do this
   for us.  */
#include "config.h"
#ifdef HAVE_CC_INHIBIT_LOOP_TO_LIBCALL
# define inhibit_loop_to_libcall \
    __attribute__ ((__optimize__ ("-fno-tree-loop-distribute-patterns")))
#else
# define inhibit_loop_to_libcall
#endif

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <unistd.h>
#include <fcntl.h>
#include <error.h>
#include <errno.h>
#include <time.h>
#include <ifunc-impl-list.h>
#define GL(x) _##x
#define GLRO(x) _##x


# define TEST_FUNCTION test_main
# define TIMEOUT (4 * 60)
# define OPT_ITERATIONS 10000
# define OPT_RANDOM 10001
# define OPT_SEED 10002

unsigned char *buf1, *buf2;
int ret, do_srandom;
unsigned int seed;
size_t page_size;

# ifndef ITERATIONS
size_t iterations = 100000;
#  define ITERATIONS_OPTIONS \
  { "iterations", required_argument, NULL, OPT_ITERATIONS },
#  define ITERATIONS_PROCESS \
  case OPT_ITERATIONS:				\
    iterations = strtoul (optarg, NULL, 0);	\
    break;
#  define ITERATIONS iterations
# else
#  define ITERATIONS_OPTIONS
#  define ITERATIONS_PROCESS
# endif

# define CMDLINE_OPTIONS ITERATIONS_OPTIONS \
  { "random", no_argument, NULL, OPT_RANDOM },	\
  { "seed", required_argument, NULL, OPT_SEED },

static void __attribute__ ((used))
cmdline_process_function (int c)
{
  switch (c)
    {
      ITERATIONS_PROCESS
      case OPT_RANDOM:
        {
          int fdr = open ("/dev/urandom", O_RDONLY);
          if (fdr < 0 || read (fdr, &seed, sizeof (seed)) != sizeof (seed))
            seed = time (NULL);
          if (fdr >= 0)
            close (fdr);
          do_srandom = 1;
          break;
        }

      case OPT_SEED:
        seed = strtoul (optarg, NULL, 0);
        do_srandom = 1;
        break;
    }
}
# define CMDLINE_PROCESS cmdline_process_function

#define CALL(impl, ...)	\
  (* (proto_t) (impl)->fn) (__VA_ARGS__)

#ifdef TEST_NAME
/* Increase size of FUNC_LIST if assert is triggered at run-time.  */
static struct libc_ifunc_impl func_list[32];
static int func_count;
static int impl_count = -1;
static impl_t *impl_array;

# define FOR_EACH_IMPL(impl, notall) \
  impl_t *impl;								\
  int count;								\
  if (impl_count == -1)							\
    {									\
      impl_count = 0;							\
      if (func_count != 0)						\
	{								\
	  int f;							\
	  impl_t *skip = NULL, *a;					\
	  for (impl = __start_impls; impl < __stop_impls; ++impl)	\
	    if (strcmp (impl->name, TEST_NAME) == 0)			\
	      skip = impl;						\
	    else							\
	      impl_count++;						\
	  a = impl_array = xmalloc ((impl_count + func_count) *		\
				    sizeof (impl_t));			\
	  for (impl = __start_impls; impl < __stop_impls; ++impl)	\
	    if (impl != skip)						\
	      *a++ = *impl;						\
	  for (f = 0; f < func_count; f++)				\
	    if (func_list[f].usable)					\
	      {								\
		a->name = func_list[f].name;				\
		a->fn = func_list[f].fn;				\
		a->test = 1;						\
		a++;							\
	      }								\
	  impl_count = a - impl_array;					\
	}								\
      else								\
        {								\
	  impl_count = __stop_impls - __start_impls;			\
	  impl_array = __start_impls;					\
        }								\
    }									\
  impl = impl_array;							\
  for (count = 0; count < impl_count; ++count, ++impl)			\
    if (!notall || impl->test)
#else
# define FOR_EACH_IMPL(impl, notall) \
  for (impl_t *impl = __start_impls; impl < __stop_impls; ++impl)	\
    if (!notall || impl->test)
#endif

#ifndef BUF1PAGES
# define BUF1PAGES 1
#endif

static void
test_init (void)
{
#ifdef TEST_NAME
  func_count = __libc_ifunc_impl_list (TEST_NAME, func_list,
				       (sizeof func_list
					/ sizeof func_list[0]));
#endif

  page_size = 2 * getpagesize ();
#ifdef MIN_PAGE_SIZE
  if (page_size < MIN_PAGE_SIZE)
    page_size = MIN_PAGE_SIZE;
#endif
  buf1 = mmap (0, (BUF1PAGES + 1) * page_size, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (buf1 == MAP_FAILED)
    error (EXIT_FAILURE, errno, "mmap failed");
  if (mprotect (buf1 + BUF1PAGES * page_size, page_size, PROT_NONE))
    error (EXIT_FAILURE, errno, "mprotect failed");
  buf2 = mmap (0, 2 * page_size, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (buf2 == MAP_FAILED)
    error (EXIT_FAILURE, errno, "mmap failed");
  if (mprotect (buf2 + page_size, page_size, PROT_NONE))
    error (EXIT_FAILURE, errno, "mprotect failed");
  if (do_srandom)
    {
      printf ("Setting seed to 0x%x\n", seed);
      srandom (seed);
    }

  memset (buf1, 0xa5, BUF1PAGES * page_size);
  memset (buf2, 0x5a, page_size);
}

#endif
