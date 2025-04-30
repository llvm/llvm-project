/* Measure string and memory functions.
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

#include <getopt.h>
#include <sys/cdefs.h>
#include <programs/xmalloc.h>

/* We are compiled under _ISOMAC, so libc-symbols.h does not do this
   for us.  */
#include "config.h"
#ifdef HAVE_CC_INHIBIT_LOOP_TO_LIBCALL
# define inhibit_loop_to_libcall \
    __attribute__ ((__optimize__ ("-fno-tree-loop-distribute-patterns")))
#else
# define inhibit_loop_to_libcall
#endif

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

# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif

# undef __USE_STRING_INLINES

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/mman.h>
# include <sys/param.h>
# include <unistd.h>
# include <fcntl.h>
# include <error.h>
# include <errno.h>
# include <time.h>
# include <ifunc-impl-list.h>
# define GL(x) _##x
# define GLRO(x) _##x
# include "bench-timing.h"

# ifndef WIDE
#  define CHAR char
#  define UCHAR unsigned char
#  define CHARBYTES 1
#  define MAX_CHAR CHAR_MAX
#  define MEMCHR memchr
#  define MEMCMP memcmp
#  define MEMCPY memcpy
#  define MEMSET memset
#  define STRCAT strcat
#  define STRLEN strlen
#  define STRCMP strcmp
#  define STRCHR strchr
#  define STRCPY strcpy
#  define STRNLEN strnlen
#  define STRCSPN strcspn
#  define STRNCAT strncat
#  define STRNCMP strncmp
#  define STRNCPY strncpy
#  define STRPBRK strpbrk
#  define STRRCHR strrchr
#  define STRSPN strspn
#  define STPCPY stpcpy
#  define STPNCPY stpncpy
# else
#  include <wchar.h>
#  define CHAR wchar_t
#  define UCHAR wchar_t
#  define CHARBYTES 4
#  define MAX_CHAR WCHAR_MAX
#  define MEMCHR wmemchr
#  define MEMCMP wmemcmp
#  define MEMCPY wmemcpy
#  define MEMSET wmemset
#  define STRCAT wcscat
#  define STRLEN wcslen
#  define STRCMP wcscmp
#  define STRCHR wcschr
#  define STRCPY wcscpy
#  define STRNLEN wcsnlen
#  define STRCSPN wcscspn
#  define STRNCAT wcsncat
#  define STRNCMP wcsncmp
#  define STRNCPY wcsncpy
#  define STRPBRK wcspbrk
#  define STRRCHR wcsrchr
#  define STRSPN wcsspn
#  define STPCPY wcpcpy
#  define STPNCPY wcpncpy
# endif /* WIDE */

# define TEST_FUNCTION test_main
# ifndef TIMEOUT
#  define TIMEOUT (4 * 60)
# endif
# define OPT_ITERATIONS 10000
# define OPT_RANDOM 10001
# define OPT_SEED 10002

# define INNER_LOOP_ITERS 8192
# define INNER_LOOP_ITERS8 32768
# define INNER_LOOP_ITERS_LARGE 131072
# define INNER_LOOP_ITERS_MEDIUM 2048
# define INNER_LOOP_ITERS_SMALL 256

int ret, do_srandom;
unsigned int seed;

# ifndef ITERATIONS
size_t iterations = 100000;
#  define ITERATIONS_OPTIONS \
     { "iterations", required_argument, NULL, OPT_ITERATIONS },
#  define ITERATIONS_PROCESS \
     case OPT_ITERATIONS:						      \
       iterations = strtoul (optarg, NULL, 0);				      \
       break;
#  define ITERATIONS iterations
# else
#  define ITERATIONS_OPTIONS
#  define ITERATIONS_PROCESS
# endif

# define CMDLINE_OPTIONS ITERATIONS_OPTIONS \
    { "random", no_argument, NULL, OPT_RANDOM },			      \
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
# define CALL(impl, ...)	\
    (* (proto_t) (impl)->fn) (__VA_ARGS__)

# ifdef TEST_NAME
/* Increase size of FUNC_LIST if assert is triggered at run-time.  */
static struct libc_ifunc_impl func_list[32];
static int func_count;
static int impl_count = -1;
static impl_t *impl_array;

#  define FOR_EACH_IMPL(impl, notall) \
     impl_t *impl;							      \
     int count;								      \
     if (impl_count == -1)						      \
       {								      \
	 impl_count = 0;						      \
	 if (func_count != 0)						      \
	   {								      \
	     int f;							      \
	     impl_t *skip = NULL, *a;					      \
	     for (impl = __start_impls; impl < __stop_impls; ++impl)	      \
	       if (strcmp (impl->name, TEST_NAME) == 0)			      \
		 skip = impl;						      \
	       else							      \
		 impl_count++;						      \
	     a = impl_array = xmalloc ((impl_count + func_count) *	      \
				       sizeof (impl_t));		      \
	     for (impl = __start_impls; impl < __stop_impls; ++impl)	      \
	       if (impl != skip)					      \
		 *a++ = *impl;						      \
	     for (f = 0; f < func_count; f++)				      \
	       if (func_list[f].usable)					      \
		 {							      \
		   a->name = func_list[f].name;				      \
		   a->fn = func_list[f].fn;				      \
		   a->test = 1;						      \
		   a++;							      \
		 }							      \
	     impl_count = a - impl_array;				      \
	   }								      \
	 else								      \
	   {								      \
	     impl_count = __stop_impls - __start_impls;			      \
	     impl_array = __start_impls;				      \
	   }								      \
       }								      \
     impl = impl_array;							      \
     for (count = 0; count < impl_count; ++count, ++impl)		      \
       if (!notall || impl->test)
# else /* !TEST_NAME */
#  define FOR_EACH_IMPL(impl, notall) \
     for (impl_t *impl = __start_impls; impl < __stop_impls; ++impl)	      \
       if (!notall || impl->test)
# endif /* !TEST_NAME */

# ifndef BUF1PAGES
#  define BUF1PAGES 1
# endif

unsigned char *buf1, *buf2;
static size_t buf1_size, buf2_size, page_size;

static void
init_sizes (void)
{
  page_size = 2 * getpagesize ();
# ifdef MIN_PAGE_SIZE
  if (page_size < MIN_PAGE_SIZE)
    page_size = MIN_PAGE_SIZE;
# endif

  buf1_size = BUF1PAGES * page_size;
  buf2_size = page_size;
}

static void
exit_error (const char *id, const char *func)
{
  error (EXIT_FAILURE, errno, "%s: %s failed", id, func);
}

/* Allocate a buffer of size SIZE with a guard page at the end.  */
static void
alloc_buf (const char *id, size_t size, unsigned char **retbuf)
{
  size_t alloc_size = size + page_size;

  if (*retbuf != NULL)
    {
	int ret = munmap (*retbuf, alloc_size);
	if (ret != 0)
	  exit_error (id, "munmap");
    }

  unsigned char *buf = mmap (0, alloc_size, PROT_READ | PROT_WRITE,
			     MAP_PRIVATE | MAP_ANON, -1, 0);

  if (buf == MAP_FAILED)
    exit_error (id, "mmap");
  if (mprotect (buf + size, page_size, PROT_NONE))
    exit_error (id, "mprotect");

  *retbuf = buf;
}

static void
alloc_bufs (void)
{
  alloc_buf ("buf1", buf1_size, &buf1);
  alloc_buf ("buf2", buf2_size, &buf2);
}

static void
test_init (void)
{
# ifdef TEST_NAME
  func_count = __libc_ifunc_impl_list (TEST_NAME, func_list,
				       (sizeof func_list
					/ sizeof func_list[0]));
# endif

  init_sizes ();
  alloc_bufs ();

  if (do_srandom)
    {
      printf ("Setting seed to 0x%x\n", seed);
      srandom (seed);
    }
}

#endif /* TEST_MAIN */
