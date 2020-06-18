/*
 * Armv7-A specific checksum implementation using NEON
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "networking.h"
#include "../chksum_common.h"

#ifndef __ARM_NEON
#pragma GCC target("+simd")
#endif

#include <arm_neon.h>

unsigned short
__chksum_arm_simd(const void *ptr, unsigned int nbytes)
{
    bool swap = (uintptr_t) ptr & 1;
    uint64x1_t vsum = { 0 };

    if (unlikely(nbytes < 40))
    {
	uint64_t sum = slurp_small(ptr, nbytes);
	return fold_and_swap(sum, false);
    }

    /* 8-byte align pointer */
    /* Inline slurp_head-like code since we use NEON here */
    Assert(nbytes >= 8);
    uint32_t off = (uintptr_t) ptr & 7;
    if (likely(off != 0))
    {
	const uint64_t *may_alias ptr64 = align_ptr(ptr, 8);
	uint64x1_t vword64 = vld1_u64(ptr64);
	/* Get rid of bytes 0..off-1 */
	uint64x1_t vmask = vdup_n_u64(ALL_ONES);
	int64x1_t vshiftl = vdup_n_s64(CHAR_BIT * off);
	vmask = vshl_u64(vmask, vshiftl);
	vword64 = vand_u64(vword64, vmask);
	uint32x2_t vtmp = vreinterpret_u32_u64(vword64);
	/* Set accumulator */
	vsum = vpaddl_u32(vtmp);
	/* Update pointer and remaining size */
	ptr = (char *) ptr64 + 8;
	nbytes -= 8 - off;
    }
    Assert(((uintptr_t) ptr & 7) == 0);

    /* Sum groups of 64 bytes */
    uint64x2_t vsum0 = { 0, 0 };
    uint64x2_t vsum1 = { 0, 0 };
    uint64x2_t vsum2 = { 0, 0 };
    uint64x2_t vsum3 = { 0, 0 };
    const uint32_t *may_alias ptr32 = ptr;
    for (uint32_t i = 0; i < nbytes / 64; i++)
    {
	uint32x4_t vtmp0 = vld1q_u32(ptr32);
	uint32x4_t vtmp1 = vld1q_u32(ptr32 + 4);
	uint32x4_t vtmp2 = vld1q_u32(ptr32 + 8);
	uint32x4_t vtmp3 = vld1q_u32(ptr32 + 12);
	vsum0 = vpadalq_u32(vsum0, vtmp0);
	vsum1 = vpadalq_u32(vsum1, vtmp1);
	vsum2 = vpadalq_u32(vsum2, vtmp2);
	vsum3 = vpadalq_u32(vsum3, vtmp3);
	ptr32 += 16;
    }
    nbytes %= 64;

    /* Fold vsum1/vsum2/vsum3 into vsum0 */
    vsum0 = vpadalq_u32(vsum0, vreinterpretq_u32_u64(vsum2));
    vsum1 = vpadalq_u32(vsum1, vreinterpretq_u32_u64(vsum3));
    vsum0 = vpadalq_u32(vsum0, vreinterpretq_u32_u64(vsum1));

    /* Add any trailing 16-byte groups */
    while (likely(nbytes >= 16))
    {
	uint32x4_t vtmp0 = vld1q_u32(ptr32);
	vsum0 = vpadalq_u32(vsum0, vtmp0);
	ptr32 += 4;
	nbytes -= 16;
    }
    Assert(nbytes < 16);

    /* Fold vsum0 into vsum */
    {
	/* 4xu32 (4x32b) -> 2xu64 (2x33b) */
	vsum0 = vpaddlq_u32(vreinterpretq_u32_u64(vsum0));
	/* 4xu32 (2x(1b+32b)) -> 2xu64 (2x(0b+32b)) */
	vsum0 = vpaddlq_u32(vreinterpretq_u32_u64(vsum0));
	/* 4xu32 (4x32b) -> 2xu64 (2x33b) */
	Assert((vgetq_lane_u64(vsum0, 0) >> 32) == 0);
	Assert((vgetq_lane_u64(vsum0, 1) >> 32) == 0);
	uint32x2_t vtmp = vmovn_u64(vsum0);
	/* Add to accumulator */
	vsum = vpadal_u32(vsum, vtmp);
    }

    /* Add any trailing group of 8 bytes */
    if (nbytes & 8)
    {
	uint32x2_t vtmp = vld1_u32(ptr32);
	/* Add to accumulator */
	vsum = vpadal_u32(vsum, vtmp);
	ptr32 += 2;
	nbytes -= 8;
    }
    Assert(nbytes < 8);

    /* Handle any trailing 1..7 bytes */
    if (likely(nbytes != 0))
    {
	Assert(((uintptr_t) ptr32 & 7) == 0);
	Assert(nbytes < 8);
	uint64x1_t vword64 = vld1_u64((const uint64_t *) ptr32);
	/* Get rid of bytes 7..nbytes */
	uint64x1_t vmask = vdup_n_u64(ALL_ONES);
	int64x1_t vshiftr = vdup_n_s64(-CHAR_BIT * (8 - nbytes));
	vmask = vshl_u64(vmask, vshiftr);/* Shift right */
	vword64 = vand_u64(vword64, vmask);
	/* Fold 64-bit sum to 33 bits */
	vword64 = vpaddl_u32(vreinterpret_u32_u64(vword64));
	/* Add to accumulator */
	vsum = vpadal_u32(vsum, vreinterpret_u32_u64(vword64));
    }

    /* Fold 64-bit vsum to 32 bits */
    vsum = vpaddl_u32(vreinterpret_u32_u64(vsum));
    vsum = vpaddl_u32(vreinterpret_u32_u64(vsum));
    Assert(vget_lane_u32(vreinterpret_u32_u64(vsum), 1) == 0);

    /* Fold 32-bit vsum to 16 bits */
    uint32x2_t vsum32 = vreinterpret_u32_u64(vsum);
    vsum32 = vpaddl_u16(vreinterpret_u16_u32(vsum32));
    vsum32 = vpaddl_u16(vreinterpret_u16_u32(vsum32));
    Assert(vget_lane_u16(vreinterpret_u16_u32(vsum32), 1) == 0);
    Assert(vget_lane_u16(vreinterpret_u16_u32(vsum32), 2) == 0);
    Assert(vget_lane_u16(vreinterpret_u16_u32(vsum32), 3) == 0);

    /* Convert to 16-bit scalar */
    uint16_t sum = vget_lane_u16(vreinterpret_u16_u32(vsum32), 0);

    if (unlikely(swap))/* Odd base pointer is unexpected */
    {
	sum = bswap16(sum);
    }
    return sum;
}
