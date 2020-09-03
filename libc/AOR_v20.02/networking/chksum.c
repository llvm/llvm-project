/*
 * Compute 16-bit sum in ones' complement arithmetic (with end-around carry).
 * This sum is often used as a simple checksum in networking.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "networking.h"
#include "chksum_common.h"

always_inline
static inline uint32_t
slurp_head32(const void **pptr, uint32_t *nbytes)
{
    uint32_t sum = 0;
    Assert(*nbytes >= 4);
    uint32_t off = (uintptr_t) *pptr % 4;
    if (likely(off != 0))
    {
	/* Get rid of bytes 0..off-1 */
	const unsigned char *ptr32 = align_ptr(*pptr, 4);
	uint32_t mask = ~0U << (CHAR_BIT * off);
	sum = load32(ptr32) & mask;
	*pptr = ptr32 + 4;
	*nbytes -= 4 - off;
    }
    return sum;
}

/* Additional loop unrolling would help when not auto-vectorizing */
unsigned short
__chksum(const void *ptr, unsigned int nbytes)
{
    bool swap = false;
    uint64_t sum = 0;

    if (nbytes > 300)
    {
	/* 4-byte align pointer */
	swap = (uintptr_t) ptr & 1;
	sum = slurp_head32(&ptr, &nbytes);
    }
    /* Else benefit of aligning not worth the overhead */

    /* Sum all 16-byte chunks */
    const char *cptr = ptr;
    for (uint32_t nquads = nbytes / 16; nquads != 0; nquads--)
    {
	uint64_t h0 = load32(cptr + 0);
	uint64_t h1 = load32(cptr + 4);
	uint64_t h2 = load32(cptr + 8);
	uint64_t h3 = load32(cptr + 12);
	sum += h0 + h1 + h2 + h3;
	cptr += 16;
    }
    nbytes %= 16;
    Assert(nbytes < 16);

    /* Handle any trailing 4-byte chunks */
    while (nbytes >= 4)
    {
	sum += load32(cptr);
	cptr += 4;
	nbytes -= 4;
    }
    Assert(nbytes < 4);

    if (nbytes & 2)
    {
	sum += load16(cptr);
	cptr += 2;
    }

    if (nbytes & 1)
    {
	sum += *(uint8_t *)cptr;
    }

    return fold_and_swap(sum, swap);
}
