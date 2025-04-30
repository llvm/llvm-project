/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "float128.h"
#include "memops.h"

#if !defined(INLINE_MEMOPS)
void
__c_mset16(float128_t *dest, float128_t value, long cnt)
{
  long i;

  for (i = 0; i < cnt; i++) {
    dest[i] = value;
  }
  return;
}
#endif
