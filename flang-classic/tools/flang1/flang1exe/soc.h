/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  soc.h - definitions/declarations for storage overlap chains. */

typedef struct {
  int sptr;
  int next;
} SOC_ITEM;

typedef struct {
  INT size;
  SOC_ITEM *base;
  INT avail;
} SOC;

extern SOC soc;

#define SOC_SPTR(i) (soc.base[i].sptr)
#define SOC_NEXT(i) (soc.base[i].next)
