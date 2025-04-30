/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>

typedef unsigned long word;

static inline word
ldq_u(const void *s)
{
  return *(const word *)((word)s & -8);
}

#define unlikely(X)	__builtin_expect ((X), 0)
#define prefetch(X)	__builtin_prefetch ((void *)(X), 0)

#define cmpbeq0(X)	__builtin_alpha_cmpbge(0, (X))
#define find(X, Y)	cmpbeq0 ((X) ^ (Y))

/* Search no more than N bytes of S for C.  */

void *
__memchr (const void *s, int xc, size_t n)
{
  const word *s_align;
  word t, current, found, mask, offset;

  if (unlikely (n == 0))
    return 0;

  current = ldq_u (s);

  /* Replicate low byte of XC into all bytes of C.  */
  t = xc & 0xff;			/* 0000000c */
  t = (t << 8) | t;			/* 000000cc */
  t = (t << 16) | t;			/* 0000cccc */
  const word c = (t << 32) | t;		/* cccccccc */

  /* Align the source, and decrement the count by the number
     of bytes searched in the first word.  */
  s_align = (const word *)((word)s & -8);
  {
    size_t inc = n + ((word)s & 7);
    n = inc | -(inc < n);
  }

  /* Deal with misalignment in the first word for the comparison.  */
  mask = (1ul << ((word)s & 7)) - 1;

  /* If the entire string fits within one word, we may need masking
     at both the front and the back of the string.  */
  if (unlikely (n <= 8))
    {
      mask |= -1ul << n;
      goto last_quad;
    }

  found = find (current, c) & ~mask;
  if (unlikely (found))
    goto found_it;

  s_align++;
  n -= 8;

  /* If the block is sufficiently large, align to cacheline and prefetch.  */
  if (unlikely (n >= 256))
    {
      /* Prefetch 3 cache lines beyond the one we're working on.  */
      prefetch (s_align + 8);
      prefetch (s_align + 16);
      prefetch (s_align + 24);

      while ((word)s_align & 63)
	{
	  current = *s_align;
	  found = find (current, c);
	  if (found)
	    goto found_it;
	  s_align++;
	  n -= 8;
	}

	/* Within each cacheline, advance the load for the next word
	   before the test for the previous word is complete.  This
	   allows us to hide the 3 cycle L1 cache load latency.  We
	   only perform this advance load within a cacheline to prevent
	   reading across page boundary.  */
#define CACHELINE_LOOP				\
	do {					\
	  word i, next = s_align[0];		\
	  for (i = 0; i < 7; ++i)		\
	    {					\
	      current = next;			\
	      next = s_align[1];		\
	      found = find (current, c);	\
	      if (unlikely (found))		\
		goto found_it;			\
	      s_align++;			\
	    }					\
	  current = next;			\
	  found = find (current, c);		\
	  if (unlikely (found))			\
	    goto found_it;			\
	  s_align++;				\
	  n -= 64;				\
	} while (0)

      /* While there's still lots more data to potentially be read,
	 continue issuing prefetches for the 4th cacheline out.  */
      while (n >= 256)
	{
	  prefetch (s_align + 24);
	  CACHELINE_LOOP;
	}

      /* Up to 3 cache lines remaining.  Continue issuing advanced
	 loads, but stop prefetching.  */
      while (n >= 64)
	CACHELINE_LOOP;

      /* We may have exhausted the buffer.  */
      if (n == 0)
	return NULL;
    }

  /* Quadword aligned loop.  */
  current = *s_align;
  while (n > 8)
    {
      found = find (current, c);
      if (unlikely (found))
	goto found_it;
      current = *++s_align;
      n -= 8;
    }

  /* The last word may need masking at the tail of the compare.  */
  mask = -1ul << n;
 last_quad:
  found = find (current, c) & ~mask;
  if (found == 0)
    return NULL;

 found_it:
#ifdef __alpha_cix__
  offset = __builtin_alpha_cttz (found);
#else
  /* Extract LSB.  */
  found &= -found;

  /* Binary search for the LSB.  */
  offset  = (found & 0x0f ? 0 : 4);
  offset += (found & 0x33 ? 0 : 2);
  offset += (found & 0x55 ? 0 : 1);
#endif

  return (void *)((word)s_align + offset);
}

#ifdef weak_alias
weak_alias (__memchr, memchr)
#endif
libc_hidden_builtin_def (memchr)
