// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunstfti
// UNSUPPORTED: mips

#include <stdio.h>


#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"
#include "int_lib.h"

// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: long double is a 128 bit floating point type
//             tu_int is a 128 bit integral type
//             value in long double is representable in tu_int or is negative 
//                 (no range checking performed)

COMPILER_RT_ABI tu_int __fixunstfti(long double a);

int test__fixunstfti(long double a, tu_int expected)
{
    tu_int x = __fixunstfti(a);
    if (x != expected)
    {
        twords xt;
        xt.all = x;

        twords expectedt;
        expectedt.all = expected;

        printf("error in __fixunstfti(%.20Lf) = 0x%.16llX%.16llX, "
               "expected 0x%.16llX%.16llX\n",
               a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(tu_int) == 4*sizeof(su_int)] = {0};
char assumption_2[sizeof(tu_int)*CHAR_BIT == 128] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__fixunstfti(makeInf128(), make_ti(0xffffffffffffffffLL,
                                               0xffffffffffffffffLL)))
        return 1;

    if (test__fixunstfti(0.0, 0))
        return 1;

    if (test__fixunstfti(0.5, 0))
        return 1;
    if (test__fixunstfti(0.99, 0))
        return 1;
    if (test__fixunstfti(1.0, 1))
        return 1;
    if (test__fixunstfti(1.5, 1))
        return 1;
    if (test__fixunstfti(1.99, 1))
        return 1;
    if (test__fixunstfti(2.0, 2))
        return 1;
    if (test__fixunstfti(2.01, 2))
        return 1;
    if (test__fixunstfti(-0.01, 0))
        return 1;
    if (test__fixunstfti(-0.99, 0))
        return 1;

    if (test__fixunstfti(0x1.p+128, make_ti(0xffffffffffffffffLL,
                                            0xffffffffffffffffLL)))
        return 1;

    if (test__fixunstfti(0x1.FFFFFEp+126, make_ti(0x7fffff8000000000LL, 0x0)))
        return 1;
    if (test__fixunstfti(0x1.FFFFFEp+127, make_ti(0xffffff0000000000LL, 0x0)))
        return 1;
    if (test__fixunstfti(0x1.FFFFFEp+128, make_ti(0xffffffffffffffffLL,
                                                  0xffffffffffffffffLL)))
        return 1;
    if (test__fixunstfti(0x1.FFFFFEp+129, make_ti(0xffffffffffffffffLL,
                                                  0xffffffffffffffffLL)))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}
