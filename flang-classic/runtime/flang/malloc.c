/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"

#include <memory.h>

#include "fort_vars.h"

extern void *shmalloc(size_t);

/* ==================== local heap routines ====================== */

#define ZIP ((char *)15L)

/* malloc */

void *
__fort_malloc_without_abort(size_t n)
{
  char *p;

  if (n == 0)
    return ZIP;
  p = malloc(n);
  if (__fort_zmem && (p != NULL))
    memset(p, '\0', n);
  return p;
}

void *
__fort_malloc(size_t n)
{
  char *p;

  p = __fort_malloc_without_abort(n);
  if (p == (char *)0)
    __fort_abort("__fort_malloc: not enough memory");
  return p;
}

/* realloc */

void *
__fort_realloc(void *ptr, size_t n)
{
  char *p;

  if (ptr == (char *)0 | ptr == ZIP) {
    if (n == 0)
      return ZIP;
    p = malloc(n);
    if (__fort_zmem && (p != NULL))
      memset(p, '\0', n);
  } else {
    if (n == 0) {
      free(ptr);
      return ZIP;
    }
    p = realloc(ptr, n);
  }
  if (p == (char *)0) {
    __fort_abort("__fort_realloc: not enough memory");
  }
  return (p);
}

/* calloc */

void *
__fort_calloc_without_abort(size_t n)
{
  char *p;

  if (n == 0)
    return ZIP;
  p = malloc(n);
  if (p != NULL)
    memset(p, '\0', n);
  return p;
}

void *
__fort_calloc(size_t n, size_t s)
{
  char *p;

  if (n == 0 | s == 0)
    return ZIP;
  p = calloc(n, s);
  if (p == (char *)0) {
    __fort_abort("__fort_calloc: not enough memory");
  }
  return (p);
}

/* free */

void
__fort_free(void *ptr)
{
  if (ptr != (char *)0 & ptr != ZIP) {
    free(ptr);
  }
}

/* ================= pseudo-global heap routines ================= */

/* stubs for global shared memory (mmapped) allocation calls */

void *
__fort_gmalloc_without_abort(size_t n)
{
  return __fort_malloc_without_abort(n);
}

void *
__fort_gmalloc(size_t n)
{
  return __fort_malloc(n);
}

void *
__fort_grealloc(void *ptr, size_t n)
{
  return __fort_realloc(ptr, n);
}

void *
__fort_gcalloc_without_abort(size_t n)
{
  return __fort_calloc_without_abort(n);
}

void *
__fort_gcalloc(size_t n, size_t s)
{
  return __fort_calloc(n, s);
}

void
__fort_gfree(void *ptr)
{
  __fort_free(ptr);
}

