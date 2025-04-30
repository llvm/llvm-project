/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* this is the dummy version of xfer_heap.c */

#include "fioMacros.h"

extern char *sbrk(int);

/* sbrk */

char *
__fort_sbrk(int len)
{
#if !defined(_WIN64)
  return (sbrk(len));
#endif
}

/* verify block is in global heap */

void
__fort_verghp(char *adr, int len, char *msg)
{
}

/* init global heap comm */

void
__fort_hinit(void)
{
}

/* send */

int
__fort_hsend(int cpu, struct ents *e)
{
  return (0);
}

/* recv */

int
__fort_hrecv(int cpu, struct ents *e)
{
  return (0);
}
