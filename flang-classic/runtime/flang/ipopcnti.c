/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

int32_t
__mth_i_ipopcnti(uint32_t u32, int size)
{
  uint32_t r32 = u32;
#if !defined(TARGET_X8664) && !defined(TARGET_LINUX_POWER)
  static const uint32_t u5s = 0x55555555;
  static const uint32_t u3s = 0x33333333;
  static const uint32_t u7s = 0x07070707;
  static const uint32_t u1s = 0x01010101;
#endif

  r32 = u32;
  if (size == 2) {
    r32 &= 0xffff;
  } else if (size == 1) {
    r32 &= 0xff;        // Slight inefficiency - don't need the u7s
  }

#if     defined(TARGET_X8664)
    asm("popcnt %1, %0"
       : "=r"(r32)
       : "r"(r32)
       );
#elif   defined(TARGET_LINUX_POWER)
    asm("popcntw    %0, %1"
       : "=r"(r32)
       : "r"(r32)
       );
#else
  r32 = (r32 & u5s) + (r32 >> 1 & u5s);
  r32 = (r32 & u3s) + (r32 >> 2 & u3s);
  r32 = (r32 & u7s) + (r32 >> 4 & u7s);
  r32 *= u1s;
  r32 >>= 24;
#endif

  return r32;

}
