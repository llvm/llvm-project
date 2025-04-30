/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int32_t
__mth_i_ipopcnt(uint32_t u32)
{
  uint32_t r32;

#if     defined(TARGET_X8664)
    asm("popcnt %1, %0"
       : "=r"(r32)
       : "r"(u32)
       );
#elif   defined(TARGET_LINUX_POWER)
    asm("popcntw    %0, %1"
       : "=r"(r32)
       : "r"(u32)
       );
#else
  static const uint32_t u5s = 0x55555555;
  static const uint32_t u3s = 0x33333333;
  static const uint32_t u7s = 0x07070707;
  static const uint32_t u1s = 0x01010101;

  r32 = u32;
  r32 = (r32 & u5s) + (r32 >> 1 & u5s);
  r32 = (r32 & u3s) + (r32 >> 2 & u3s);
  r32 = (r32 & u7s) + (r32 >> 4 & u7s);
  r32 *= u1s;
  r32 >>= 24;
#endif

  return r32;
}
