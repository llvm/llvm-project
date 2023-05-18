/*
 * random.c - random number generator for producing mathlib test cases
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "types.h"
#include "random.h"

static uint32 seedbuf[55];
static int seedptr;

void seed_random(uint32 seed) {
    int i;

    seedptr = 0;
    for (i = 0; i < 55; i++) {
        seed = seed % 44488 * 48271 - seed / 44488 * 3399;
        seedbuf[i] = seed - 1;
    }
}

uint32 base_random(void) {
    seedptr %= 55;
    seedbuf[seedptr] += seedbuf[(seedptr+31)%55];
    return seedbuf[seedptr++];
}

uint32 random32(void) {
    uint32 a, b, b1, b2;
    a = base_random();
    b = base_random();
    for (b1 = 0x80000000, b2 = 1; b1 > b2; b1 >>= 1, b2 <<= 1) {
        uint32 b3 = b1 | b2;
        if ((b & b3) != 0 && (b & b3) != b3)
            b ^= b3;
    }
    return a ^ b;
}

/*
 * random_upto: generate a uniformly randomised number in the range
 * 0,...,limit-1. (Precondition: limit > 0.)
 *
 * random_upto_biased: generate a number in the same range, but with
 * the probability skewed towards the high end by means of taking the
 * maximum of 8*bias+1 samples from the uniform distribution on the
 * same range. (I don't know why bias is given in that curious way -
 * historical reasons, I expect.)
 *
 * For speed, I separate the implementation of random_upto into the
 * two stages of (a) generate a bitmask which reduces a 32-bit random
 * number to within a factor of two of the right range, (b) repeatedly
 * generate numbers in that range until one is small enough. Splitting
 * it up like that means that random_upto_biased can do (a) only once
 * even when it does (b) lots of times.
 */

static uint32 random_upto_makemask(uint32 limit) {
    uint32 mask = 0xFFFFFFFF;
    int i;
    for (i = 16; i > 0; i >>= 1)
        if ((limit & (mask >> i)) == limit)
            mask >>= i;
    return mask;
}

static uint32 random_upto_internal(uint32 limit, uint32 mask) {
    uint32 ret;
    do {
        ret = random32() & mask;
    } while (ret > limit);
    return ret;
}

uint32 random_upto(uint32 limit) {
    uint32 mask = random_upto_makemask(limit);
    return random_upto_internal(limit, mask);
}

uint32 random_upto_biased(uint32 limit, int bias) {
    uint32 mask = random_upto_makemask(limit);

    uint32 ret = random_upto_internal(limit, mask);
    while (bias--) {
        uint32 tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
        tmp = random_upto_internal(limit, mask); if (tmp < ret) ret = tmp;
    }

    return ret;
}
