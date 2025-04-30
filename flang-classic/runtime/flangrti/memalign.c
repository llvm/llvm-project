/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdio.h>
#include <stdlib.h>

#if (defined(WIN32) || defined(WIN64))
extern void *_aligned_malloc(size_t, size_t);
extern void _aligned_free(void *);
#endif

void *
__aligned_malloc(size_t sz, size_t aln)
{
  char *q;
  size_t need;

/*
 * MINALN must be a multiple of sizeof(ptr) and sufficient for aligned
 * accesses
 */
#define MINALN 16

  if (!aln || aln < MINALN)
    aln = MINALN;
  else {
    /* make sure aln is a power of two */
    int s;
    s = 0;
    while ((aln & 1) == 0) {
      s++;
      aln >>= 1;
    }
    aln = 1 << s;
  }
  need = sz + MINALN;
#if (defined(WIN32) || defined(WIN64))
  q = _aligned_malloc(need, aln);
  if (!q)
    return NULL;
#else
  if (posix_memalign((void**)&q, aln, need))
    return NULL;
#endif
  return q;
}
void
__aligned_free(void *p)
{
#if (defined(WIN32) || defined(WIN64))
  _aligned_free(p);
#else
  free(p);
#endif
  return;
}

