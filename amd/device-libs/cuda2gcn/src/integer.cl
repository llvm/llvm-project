/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ATTR __attribute__((always_inline, const))

//-------- T __nv_mul24
ATTR int __nv_mul24(int x, int y) { return __ockl_mul24_i32(x, y); }

//-------- T __nv_umul24
ATTR uint __nv_umul24(uint x, uint y) { return __ockl_mul24_u32(x, y); }

//-------- T __nv_mul64hi
ATTR long __nv_mul64hi(long x, long y) { return __ockl_mul_hi_i64(x,y); }

//-------- T __nv_mulhi
ATTR int __nv_mulhi(int x, int y) { return __ockl_mul_hi_i32(x,y); }

//-------- T __nv_umul64hi
ATTR ulong __nv_umul64hi(ulong x, ulong y) { return __ockl_mul_hi_u64(x,y); }

//-------- T __nv_umulhi
ATTR uint __nv_umulhi(uint x, uint y) { return __ockl_mul_hi_u32(x,y); }

