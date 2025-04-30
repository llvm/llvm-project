/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int64_t
__mth_i_kpoppar(uint64_t u64)
{
  uint64_t r64;

#if     defined(TARGET_X8664)
    asm("popcnt %1, %0\n"
        "\tandq $0x1, %0"
       : "=r"(r64)
       : "r"(u64)
       );
#elif   defined(TARGET_LINUX_POWER)
    asm("popcntd    %0, %1\n"
        "\trldicl   %0, %0, 0, 63"
       : "=r"(r64)
       : "r"(u64)
       );
#else
  r64 = u64;
  r64 ^= r64 >> 32;
  r64 ^= r64 >> 16;
  r64 ^= r64 >> 8;
  r64 ^= r64 >> 4;
  r64 ^= r64 >> 2;
  r64 ^= r64 >> 1;
  r64 &= 0x1;
#endif

  return r64;
}
