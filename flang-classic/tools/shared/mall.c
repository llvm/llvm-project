/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  \brief customized storage allocation routines for compiler.
 */

#include "mall.h"
#include "global.h"
#include "error.h"
#if DEBUG
#include <string.h>

#define TRACE(str, d)                           \
  if (DBGBIT(7, 1))                             \
    fprintf(stderr, str, d)

void
bjunk(void *p, BIGUINT64 n)
{
  memset(p, -99, n);
}
#else
#define TRACE(a, b)
#endif

#define TOO_LARGE \
  F_0007_Subprogram_too_large_to_compile_at_this_optimization_level_OP1

char *
sccalloc(BIGUINT64 nbytes)
{
  char *p;

  TRACE("sccalloc called to get %ld bytes\n", nbytes);
  p = (char*)malloc(nbytes);
  if (p == NULL)
    errfatal(TOO_LARGE);
#if DEBUG
  if (DBGBIT(0, 0x20000)) {
    char *q, cc;
    unsigned int s;
    /* fill with junk */
    cc = 0xa6;
    for (s = nbytes, q = p; s; --s, ++q) {
      *q = cc;
      cc = (cc << 1) | (cc >> 7);
    }
  }
#endif
  TRACE("sccalloc returns %p\n", p);
  return p;
}

/*****************************************************************/

void
sccfree(char *ap)
{
  TRACE("sccfree called to free %p\n", ap);
  free(ap);
}

/**********************************************************/

char *
sccrelal(char *pp, BIGUINT64 nbytes)
{
  char *q;
  TRACE("sccrelal called to realloc %p\n", pp);
  q = (char*)realloc(pp, nbytes);
  if (q == NULL)
    errfatal(TOO_LARGE);
  TRACE("sccrelal returns %p\n", q);
  return q;
}
