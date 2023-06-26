/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef IRIF_H
#define IRIF_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define REQUIRES_16BIT_INSTS __attribute__((target("16-bit-insts")))
#define REQUIRES_WAVE32 __attribute__((target("wavefrontsize32")))
#define REQUIRES_WAVE64 __attribute__((target("wavefrontsize64")))

#define BUILTIN_CLZ_U8(x) (uchar)(x == 0u ? 8 : __builtin_clz(x) - 24)
#define BUILTIN_CLZ_U16(x) (ushort)(x == 0u ? 16 : __builtin_clzs(x))
#define BUILTIN_CLZ_U32(x) (uint)(x == 0u ? 32 : __builtin_clz(x))
#define BUILTIN_CLZ_U64(x) (ulong)(x == 0u ? 64 : __builtin_clzl(x))

#define BUILTIN_CTZ_U8(x) (uchar)(x == 0u ? (uchar)8 : __builtin_ctz((uint)x))
#define BUILTIN_CTZ_U16(x) (ushort)(x == 0u ? 16 : __builtin_ctzs(x))
#define BUILTIN_CTZ_U32(x) (uint)(x == 0u ? 32 : __builtin_ctz(x))
#define BUILTIN_CTZ_U64(x) (ulong)(x == 0u ? 64 : __builtin_ctzl(x))

#pragma OPENCL EXTENSION cl_khr_fp16 : disable
#endif // IRIF_H
