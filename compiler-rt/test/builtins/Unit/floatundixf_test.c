//===-- floatundixf_test.c - Test __floatundixf ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatundixf for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>


#if HAS_80_BIT_LONG_DOUBLE
// Returns: convert a to a long double, rounding toward even.

// Assumption: long double is a IEEE 80 bit floating point type padded to 128 bits
//             du_int is a 64 bit integral type

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee eeee |
// 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI long double __floatundixf(du_int a);

int test__floatundixf(du_int a, long double expected)
{
    long double x = __floatundixf(a);
    if (x != expected)
        printf("error in __floatundixf(%llX) = %LA, expected %LA\n",
               a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(du_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(du_int)*CHAR_BIT == 64] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};
#endif

int main()
{
#if HAS_80_BIT_LONG_DOUBLE
    if (test__floatundixf(0, 0.0))
        return 1;

    if (test__floatundixf(1, 1.0))
        return 1;
    if (test__floatundixf(2, 2.0))
        return 1;
    if (test__floatundixf(20, 20.0))
        return 1;

    if (test__floatundixf(0x7FFFFF8000000000ULL, 0x1.FFFFFEp+62))
        return 1;
    if (test__floatundixf(0x7FFFFFFFFFFFF800ULL, 0x1.FFFFFFFFFFFFEp+62))
        return 1;
    if (test__floatundixf(0x7FFFFF0000000000ULL, 0x1.FFFFFCp+62))
        return 1;
    if (test__floatundixf(0x7FFFFFFFFFFFF000ULL, 0x1.FFFFFFFFFFFFCp+62))
        return 1;
    if (test__floatundixf(0x7FFFFFFFFFFFFFFFULL, 0xF.FFFFFFFFFFFFFFEp+59L))
        return 1;
    if (test__floatundixf(0xFFFFFFFFFFFFFFFEULL, 0xF.FFFFFFFFFFFFFFEp+60L))
        return 1;
    if (test__floatundixf(0xFFFFFFFFFFFFFFFFULL, 0xF.FFFFFFFFFFFFFFFp+60L))
        return 1;

    if (test__floatundixf(0x8000008000000000ULL, 0x8.000008p+60))
        return 1;
    if (test__floatundixf(0x8000000000000800ULL, 0x8.0000000000008p+60))
        return 1;
    if (test__floatundixf(0x8000010000000000ULL, 0x8.00001p+60))
        return 1;
    if (test__floatundixf(0x8000000000001000ULL, 0x8.000000000001p+60))
        return 1;

    if (test__floatundixf(0x8000000000000000ULL, 0x8p+60))
        return 1;
    if (test__floatundixf(0x8000000000000001ULL, 0x8.000000000000001p+60L))
        return 1;

    if (test__floatundixf(0x0007FB72E8000000ULL, 0x1.FEDCBAp+50))
        return 1;

    if (test__floatundixf(0x0007FB72EA000000ULL, 0x1.FEDCBA8p+50))
        return 1;
    if (test__floatundixf(0x0007FB72EB000000ULL, 0x1.FEDCBACp+50))
        return 1;
    if (test__floatundixf(0x0007FB72EBFFFFFFULL, 0x1.FEDCBAFFFFFFCp+50))
        return 1;
    if (test__floatundixf(0x0007FB72EC000000ULL, 0x1.FEDCBBp+50))
        return 1;
    if (test__floatundixf(0x0007FB72E8000001ULL, 0x1.FEDCBA0000004p+50))
        return 1;

    if (test__floatundixf(0x0007FB72E6000000ULL, 0x1.FEDCB98p+50))
        return 1;
    if (test__floatundixf(0x0007FB72E7000000ULL, 0x1.FEDCB9Cp+50))
        return 1;
    if (test__floatundixf(0x0007FB72E7FFFFFFULL, 0x1.FEDCB9FFFFFFCp+50))
        return 1;
    if (test__floatundixf(0x0007FB72E4000001ULL, 0x1.FEDCB90000004p+50))
        return 1;
    if (test__floatundixf(0x0007FB72E4000000ULL, 0x1.FEDCB9p+50))
        return 1;

    if (test__floatundixf(0x023479FD0E092DC0ULL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floatundixf(0x023479FD0E092DA1ULL, 0x8.D1E7F43824B684p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DB0ULL, 0x8.D1E7f43824B6Cp+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DB8ULL, 0x8.D1E7F43824B6Ep+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DB6ULL, 0x8.D1E7F43824B6D8p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DBFULL, 0x8.D1E7F43824B6FCp+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DC1ULL, 0x8.D1E7F43824B704p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DC7ULL, 0x8.D1E7F43824B71Cp+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DC8ULL, 0x8.D1E7F43824B72p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DCFULL, 0x8.D1E7F43824B73Cp+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DD0ULL, 0x8.D1E7F43824B74p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DD1ULL, 0x8.D1E7F43824B744p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DD8ULL, 0x8.D1E7F43824B76p+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DDFULL, 0x8.D1E7F43824B77Cp+54L))
        return 1;
    if (test__floatundixf(0x023479FD0E092DE0ULL, 0x1.1A3CFE870496Fp+57))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}
