/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdlib.h>
#include "global.h"

int
__fort_atol(char *p)
{
  int n;
  char *q;

  if (p == (char *)0) {
    return (0);
  }
  n = strtol(p, &q, 0);
  switch (*q) {
  case 'k':
  case 'K':
    n <<= 10;
    break;
  case 'm':
  case 'M':
    n <<= 20;
    break;
  case 'g':
  case 'G':
    n <<= 30;
    break;
  }
  return (n);
}

long
__fort_strtol(const char *str, char **ptr, int base)
{
  long val;
  char *end;

  if (str) {
    val = strtol(str, &end, base);
    if (end != str) {
      switch (*end) {
      case 'g':
      case 'G':
        val <<= 10;
        FLANG_FALLTHROUGH;
      case 'm':
      case 'M':
        val <<= 10;
        FLANG_FALLTHROUGH;
      case 'k':
      case 'K':
        val <<= 10;
        ++end;
      }
    }
  } else {
    val = 0;
    end = NULL;
  }
  if (ptr)
    *ptr = end;
  return val;
}
