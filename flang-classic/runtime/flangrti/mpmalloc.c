/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* mp-safe wrappers for malloc, etc. */

#ifdef TARGET_LINUX
#include <features.h>
#endif

#include <stdlib.h>
#include "llcrit.h"
#ifndef __GNU_LIBRARY__
MP_SEMAPHORE(static, sem);
#endif

void *
_mp_malloc(size_t n)
{
  void *p;

#ifndef __GNU_LIBRARY__
  _mp_p(&sem);
#endif
  p = malloc(n);
#ifndef __GNU_LIBRARY__
  _mp_v(&sem);
#endif
  return (p);
}

void *
_mp_calloc(size_t n, size_t t)
{
  void *p;

#ifndef __GNU_LIBRARY__
  _mp_p(&sem);
#endif
  p = calloc(n, t);
#ifndef __GNU_LIBRARY__
  _mp_v(&sem);
#endif
  return (p);
}

void *
_mp_realloc(void *p, size_t n)
{
  void *q;

#ifndef __GNU_LIBRARY__
  _mp_p(&sem);
#endif
  q = realloc(p, n);
#ifndef __GNU_LIBRARY__
  _mp_v(&sem);
#endif
  return (q);
}

void
_mp_free(void *p)
{
  if (p == 0)
    return;
#ifndef __GNU_LIBRARY__
  _mp_p(&sem);
#endif
  free(p);
#ifndef __GNU_LIBRARY__
  _mp_v(&sem);
#endif
}
