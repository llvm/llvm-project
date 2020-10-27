/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_ASM_INT128_CAS_H_INCLUDED
#define ABTD_ASM_INT128_CAS_H_INCLUDED

#include <stdint.h>

static inline int ABTD_asm_bool_cas_weak_int128(__int128 *var, __int128 oldv,
                                                __int128 newv)
{
#if defined(__x86_64__)

    /*
     * Use a compare-and-swap instruction.
     * See src/lockfree/x86-64.h in https://github.com/ARM-software/progress64/
     */
    union u128_union {
        struct {
            uint64_t lo, hi;
        } s;
        __int128 ui;
    };
    union u128_union cmp, with;
    cmp.ui = oldv;
    with.ui = newv;
    char result;
    __asm__ __volatile__("lock cmpxchg16b %1\n"
                         "setz %0"
                         : "=&q"(result), "+m"(*var), "+d"(cmp.s.hi),
                           "+a"(cmp.s.lo)
                         : "c"(with.s.hi), "b"(with.s.lo)
                         : "memory", "cc");
    return !!result;

#elif defined(__aarch64__)

#if 0
    /*
     * Use a compare-and-swap instruction.
     * See src/lockfree/aarch64.h in https://github.com/ARM-software/progress64/
     *
     * I added "cc" since I am not 100% sure that these instructions do not
     * modify the status flag.  Since the following version has not been tested
     * on a real machine, it is disabled.
     */
    __int128 prev;
#if __GNUC__ >= 9
    /* This version needs further testing. */
    __asm__ __volatile__("caspal %0, %H0, %1, %H1, [%2]"
                         : "+r"(oldv)
                         : "r"(newv), "r"(var)
                         : "memory", "cc");
    prev = x0 | ((__int128)x1 << 64);
#else
    __asm__ __volatile__("" ::: "memory");
    register uint64_t x0 __asm__("x0") = (uint64_t)oldv;
    register uint64_t x1 __asm__("x1") = (uint64_t)(oldv >> 64);
    register uint64_t x2 __asm__("x2") = (uint64_t)newv;
    register uint64_t x3 __asm__("x3") = (uint64_t)(newv >> 64);
    __asm__ __volatile__("caspal x0, %[old2], %[newv1], %[newv2], [%[v]]"
                         : [old1] "+r"(x0), [old2] "+r"(x1)
                         : [newv1] "r"(x2), [newv2] "r"(x3), [v] "r"(var)
                         : "memory", "cc");
    prev = x0 | ((__int128)x1 << 64);
#endif
    return oldv == prev;
#else
    /*
     * Use exclusive load and store instructions (LL/SC).
     * See src/lockfree/aarch64.h in https://github.com/ARM-software/progress64/
     */
    __int128 prev;
    __asm__ __volatile__("ldaxp %0, %H0, [%1]"
                         : "=&r"(prev)
                         : "r"(var)
                         : "memory");
    if (prev != oldv) {
        /* Already rewritten. */
        return 0;
    }
    uint32_t ret;
    __asm__ __volatile__("stlxp %w0, %1, %H1, [%2]"
                         : "=&r"(ret)
                         : "r"(newv), "r"(var)
                         : "memory");
    return !ret;
#endif

#elif defined(__ppc64__) || defined(__PPC64__)

    /* Use "reserve-indexed" load and store (LL/SC) */
    int ret = 0;
    /* prev0 and newv0 must be even-indexed registers. */
    register volatile uint64_t prev0 __asm__("r10");
    register volatile uint64_t prev1 __asm__("r11");
    register volatile uint64_t newv0 __asm__("r8") = (newv >> 64);
    register volatile uint64_t newv1 __asm__("r9") = newv;
    uint64_t oldv0 = (oldv >> 64);
    uint64_t oldv1 = oldv;
    __asm__ __volatile__("\n"
                         "\tlwsync\n"
                         "\tlqarx %[pv0], 0, %[ptr]\n"
                         "\tcmpd %[pv0], %[ov0]\n"
                         "\tbne 1f\n"
                         "\tcmpd %[pv1], %[ov1]\n"
                         "\tbne 1f\n"
                         "\tstqcx. %[nv0], 0, %[ptr]\n"
                         "\tbne 1f\n"
                         "\tli %[ret], 1\n"
                         "1:\n"
                         "\tisync\n"
                         : [pv0] "+&r"(prev0), [pv1] "+&r"(prev1),
                           [ret] "+&r"(ret)
                         : [ptr] "r"(var), [ov0] "r"(oldv0), [ov1] "r"(oldv1),
                           [nv0] "r"(newv0), [nv1] "r"(newv1)
                         : "memory", "cc");
    return ret;

#else

#error "Argobots does not support 128-bit CAS for this architecture."

#endif
}

#endif /* ABTD_ASM_INT128_CAS_H_INCLUDED */
