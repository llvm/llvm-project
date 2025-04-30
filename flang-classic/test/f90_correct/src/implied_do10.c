/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

_Noreturn void f90_alloc04a_i8 (size_t *nelem, void *kind, size_t *len,
                                void *stat, void *pointer, void *offset,
                                void *firsttime, void *align, void *errmsg_adr,
                                size_t *errmsg_len)
{
  printf("nelem: %lu\n", *nelem);
  if (2UL < *nelem)
    abort();
  printf(" PASSED\n");
  exit(0);
}
