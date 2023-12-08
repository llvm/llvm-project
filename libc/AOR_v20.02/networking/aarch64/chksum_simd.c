/*
 * AArch64-specific checksum implementation using NEON
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

always_inline
static inline uint64_t
slurp_head64(const void **pptr, uint32_t *nbytes)
{
    Assert(*nbytes >= 8);
    uint64_t sum = 0;
    uint32_t off = (uintptr_t) *pptr % 8;
    if (likely(off != 0))
    {
	/* Get rid of bytes 0..off-1 */
	const unsigned char *ptr64 = align_ptr(*pptr, 8);
	uint64_t mask = ALL_ONES << (CHAR_BIT * off);
	uint64_t val = load64(ptr64) & mask;
	/* Fold 64-bit sum to 33 bits */
	sum = val >> 32;
	sum += (uint32_t) val;
	*pptr = ptr64 + 8;
	*nbytes -= 8 - off;
    }
    return sum;
}

always_inline
static inline uint64_t
slurp_tail64(uint64_t sum, const void *ptr, uint32_t nbytes)
{
    Assert(nbytes < 8);
    if (likely(nbytes != 0))
    {
	/* Get rid of bytes 7..nbytes */
	uint64_t mask = ALL_ONES >> (CHAR_BIT * (8 - nbytes));
	Assert(__builtin_popcountl(mask) / CHAR_BIT == nbytes);
	uint64_t val = load64(ptr) & mask;
	sum += val >> 32;
	sum += (uint32_t) val;
	nbytes = 0;
    }
    Assert(nbytes == 0);
    return sum;
}

unsigned short
__chksum_aarch64_simd(const void *ptr, unsigned int nbytes)
{
    bool swap = (uintptr_t) ptr & 1;
    uint64_t sum;

    if (unlikely(nbytes < 50))
    {
	sum = slurp_small(ptr, nbytes);
	swap = false;
	goto fold;
    }

    /* 8-byte align pointer */
    Assert(nbytes >= 8);
    sum = slurp_head64(&ptr, &nbytes);
    Assert(((uintptr_t) ptr & 7) == 0);

    const uint32_t *may_alias ptr32 = ptr;

    uint64x2_t vsum0 = { 0, 0 };
    uint64x2_t vsum1 = { 0, 0 };
    uint64x2_t vsum2 = { 0, 0 };
    uint64x2_t vsum3 = { 0, 0 };

    /* Sum groups of 64 bytes */
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

    /* Fold vsum2 and vsum3 into vsum0 and vsum1 */
    vsum0 = vpadalq_u32(vsum0, vreinterpretq_u32_u64(vsum2));
    vsum1 = vpadalq_u32(vsum1, vreinterpretq_u32_u64(vsum3));

    /* Add any trailing group of 32 bytes */
    if (nbytes & 32)
    {
	uint32x4_t vtmp0 = vld1q_u32(ptr32);
	uint32x4_t vtmp1 = vld1q_u32(ptr32 + 4);
	vsum0 = vpadalq_u32(vsum0, vtmp0);
	vsum1 = vpadalq_u32(vsum1, vtmp1);
	ptr32 += 8;
	nbytes -= 32;
    }
    Assert(nbytes < 32);

    /* Fold vsum1 into vsum0 */
    vsum0 = vpadalq_u32(vsum0, vreinterpretq_u32_u64(vsum1));

    /* Add any trailing group of 16 bytes */
    if (nbytes & 16)
    {
	uint32x4_t vtmp = vld1q_u32(ptr32);
	vsum0 = vpadalq_u32(vsum0, vtmp);
	ptr32 += 4;
	nbytes -= 16;
    }
    Assert(nbytes < 16);

    /* Add any trailing group of 8 bytes */
    if (nbytes & 8)
    {
	uint32x2_t vtmp = vld1_u32(ptr32);
	vsum0 = vaddw_u32(vsum0, vtmp);
	ptr32 += 2;
	nbytes -= 8;
    }
    Assert(nbytes < 8);

    uint64_t val = vaddlvq_u32(vreinterpretq_u32_u64(vsum0));
    sum += val >> 32;
    sum += (uint32_t) val;

    /* Handle any trailing 0..7 bytes */
    sum = slurp_tail64(sum, ptr32, nbytes);

fold:
    return fold_and_swap(sum, swap);
}
