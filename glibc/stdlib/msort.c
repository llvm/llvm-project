/* An alternative to qsort, with an identical interface.
   This file is part of the GNU C Library.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
   Written by Mike Haertel, September 1988.

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

#include <alloca.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <memcopy.h>
#include <errno.h>
#include <atomic.h>

struct msort_param
{
  size_t s;
  size_t var;
  __compar_d_fn_t cmp;
  void *arg;
  char *t;
};
static void msort_with_tmp (const struct msort_param *p, void *b, size_t n);

static void
msort_with_tmp (const struct msort_param *p, void *b, size_t n)
{
  char *b1, *b2;
  size_t n1, n2;

  if (n <= 1)
    return;

  n1 = n / 2;
  n2 = n - n1;
  b1 = b;
  b2 = (char *) b + (n1 * p->s);

  msort_with_tmp (p, b1, n1);
  msort_with_tmp (p, b2, n2);

  char *tmp = p->t;
  const size_t s = p->s;
  __compar_d_fn_t cmp = p->cmp;
  void *arg = p->arg;
  switch (p->var)
    {
    case 0:
      while (n1 > 0 && n2 > 0)
	{
	  if ((*cmp) (b1, b2, arg) <= 0)
	    {
	      *(uint32_t *) tmp = *(uint32_t *) b1;
	      b1 += sizeof (uint32_t);
	      --n1;
	    }
	  else
	    {
	      *(uint32_t *) tmp = *(uint32_t *) b2;
	      b2 += sizeof (uint32_t);
	      --n2;
	    }
	  tmp += sizeof (uint32_t);
	}
      break;
    case 1:
      while (n1 > 0 && n2 > 0)
	{
	  if ((*cmp) (b1, b2, arg) <= 0)
	    {
	      *(uint64_t *) tmp = *(uint64_t *) b1;
	      b1 += sizeof (uint64_t);
	      --n1;
	    }
	  else
	    {
	      *(uint64_t *) tmp = *(uint64_t *) b2;
	      b2 += sizeof (uint64_t);
	      --n2;
	    }
	  tmp += sizeof (uint64_t);
	}
      break;
    case 2:
      while (n1 > 0 && n2 > 0)
	{
	  unsigned long *tmpl = (unsigned long *) tmp;
	  unsigned long *bl;

	  tmp += s;
	  if ((*cmp) (b1, b2, arg) <= 0)
	    {
	      bl = (unsigned long *) b1;
	      b1 += s;
	      --n1;
	    }
	  else
	    {
	      bl = (unsigned long *) b2;
	      b2 += s;
	      --n2;
	    }
	  while (tmpl < (unsigned long *) tmp)
	    *tmpl++ = *bl++;
	}
      break;
    case 3:
      while (n1 > 0 && n2 > 0)
	{
	  if ((*cmp) (*(const void **) b1, *(const void **) b2, arg) <= 0)
	    {
	      *(void **) tmp = *(void **) b1;
	      b1 += sizeof (void *);
	      --n1;
	    }
	  else
	    {
	      *(void **) tmp = *(void **) b2;
	      b2 += sizeof (void *);
	      --n2;
	    }
	  tmp += sizeof (void *);
	}
      break;
    default:
      while (n1 > 0 && n2 > 0)
	{
	  if ((*cmp) (b1, b2, arg) <= 0)
	    {
	      tmp = (char *) __mempcpy (tmp, b1, s);
	      b1 += s;
	      --n1;
	    }
	  else
	    {
	      tmp = (char *) __mempcpy (tmp, b2, s);
	      b2 += s;
	      --n2;
	    }
	}
      break;
    }

  if (n1 > 0)
    memcpy (tmp, b1, n1 * s);
  memcpy (b, p->t, (n - n2) * s);
}


void
__qsort_r (void *b, size_t n, size_t s, __compar_d_fn_t cmp, void *arg)
{
  size_t size = n * s;
  char *tmp = NULL;
  struct msort_param p;

  /* For large object sizes use indirect sorting.  */
  if (s > 32)
    size = 2 * n * sizeof (void *) + s;

  if (size < 1024)
    /* The temporary array is small, so put it on the stack.  */
    p.t = __alloca (size);
  else
    {
      /* We should avoid allocating too much memory since this might
	 have to be backed up by swap space.  */
      static long int phys_pages;
      static int pagesize;

      if (pagesize == 0)
	{
	  phys_pages = __sysconf (_SC_PHYS_PAGES);

	  if (phys_pages == -1)
	    /* Error while determining the memory size.  So let's
	       assume there is enough memory.  Otherwise the
	       implementer should provide a complete implementation of
	       the `sysconf' function.  */
	    phys_pages = (long int) (~0ul >> 1);

	  /* The following determines that we will never use more than
	     a quarter of the physical memory.  */
	  phys_pages /= 4;

	  /* Make sure phys_pages is written to memory.  */
	  atomic_write_barrier ();

	  pagesize = __sysconf (_SC_PAGESIZE);
	}

      /* Just a comment here.  We cannot compute
	   phys_pages * pagesize
	   and compare the needed amount of memory against this value.
	   The problem is that some systems might have more physical
	   memory then can be represented with a `size_t' value (when
	   measured in bytes.  */

      /* If the memory requirements are too high don't allocate memory.  */
      if (size / pagesize > (size_t) phys_pages)
	{
	  _quicksort (b, n, s, cmp, arg);
	  return;
	}

      /* It's somewhat large, so malloc it.  */
      int save = errno;
      tmp = malloc (size);
      __set_errno (save);
      if (tmp == NULL)
	{
	  /* Couldn't get space, so use the slower algorithm
	     that doesn't need a temporary array.  */
	  _quicksort (b, n, s, cmp, arg);
	  return;
	}
      p.t = tmp;
    }

  p.s = s;
  p.var = 4;
  p.cmp = cmp;
  p.arg = arg;

  if (s > 32)
    {
      /* Indirect sorting.  */
      char *ip = (char *) b;
      void **tp = (void **) (p.t + n * sizeof (void *));
      void **t = tp;
      void *tmp_storage = (void *) (tp + n);

      while ((void *) t < tmp_storage)
	{
	  *t++ = ip;
	  ip += s;
	}
      p.s = sizeof (void *);
      p.var = 3;
      msort_with_tmp (&p, p.t + n * sizeof (void *), n);

      /* tp[0] .. tp[n - 1] is now sorted, copy around entries of
	 the original array.  Knuth vol. 3 (2nd ed.) exercise 5.2-10.  */
      char *kp;
      size_t i;
      for (i = 0, ip = (char *) b; i < n; i++, ip += s)
	if ((kp = tp[i]) != ip)
	  {
	    size_t j = i;
	    char *jp = ip;
	    memcpy (tmp_storage, ip, s);

	    do
	      {
		size_t k = (kp - (char *) b) / s;
		tp[j] = jp;
		memcpy (jp, kp, s);
		j = k;
		jp = kp;
		kp = tp[k];
	      }
	    while (kp != ip);

	    tp[j] = jp;
	    memcpy (jp, tmp_storage, s);
	  }
    }
  else
    {
      if ((s & (sizeof (uint32_t) - 1)) == 0
	  && ((char *) b - (char *) 0) % __alignof__ (uint32_t) == 0)
	{
	  if (s == sizeof (uint32_t))
	    p.var = 0;
	  else if (s == sizeof (uint64_t)
		   && ((char *) b - (char *) 0) % __alignof__ (uint64_t) == 0)
	    p.var = 1;
	  else if ((s & (sizeof (unsigned long) - 1)) == 0
		   && ((char *) b - (char *) 0)
		      % __alignof__ (unsigned long) == 0)
	    p.var = 2;
	}
      msort_with_tmp (&p, b, n);
    }
  free (tmp);
}
libc_hidden_def (__qsort_r)
weak_alias (__qsort_r, qsort_r)


void
qsort (void *b, size_t n, size_t s, __compar_fn_t cmp)
{
  return __qsort_r (b, n, s, (__compar_d_fn_t) cmp, NULL);
}
libc_hidden_def (qsort)
