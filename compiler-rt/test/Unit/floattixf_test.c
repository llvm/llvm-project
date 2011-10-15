//===-- floattixf.c - Test __floattixf ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floattixf for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#if __x86_64

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

// Returns: convert a to a long double, rounding toward even.

// Assumption: long double is a IEEE 80 bit floating point type padded to 128 bits
//             ti_int is a 128 bit integral type

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee eeee |
// 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

long double __floattixf(ti_int a);

int test__floattixf(ti_int a, long double expected)
{
    long double x = __floattixf(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        printf("error in __floattixf(0x%.16llX%.16llX) = %LA, expected %LA\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(ti_int)*CHAR_BIT == 128] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __x86_64
    if (test__floattixf(0, 0.0))
        return 1;

    if (test__floattixf(1, 1.0))
        return 1;
    if (test__floattixf(2, 2.0))
        return 1;
    if (test__floattixf(20, 20.0))
        return 1;
    if (test__floattixf(-1, -1.0))
        return 1;
    if (test__floattixf(-2, -2.0))
        return 1;
    if (test__floattixf(-20, -20.0))
        return 1;

    if (test__floattixf(0x7FFFFF8000000000LL, 0x1.FFFFFEp+62))
        return 1;
    if (test__floattixf(0x7FFFFFFFFFFFF800LL, 0x1.FFFFFFFFFFFFEp+62))
        return 1;
    if (test__floattixf(0x7FFFFF0000000000LL, 0x1.FFFFFCp+62))
        return 1;
    if (test__floattixf(0x7FFFFFFFFFFFF000LL, 0x1.FFFFFFFFFFFFCp+62))
        return 1;

    if (test__floattixf(make_ti(0x8000008000000000LL, 0), -0x1.FFFFFEp+126))
        return 1;
    if (test__floattixf(make_ti(0x8000000000000800LL, 0), -0x1.FFFFFFFFFFFFEp+126))
        return 1;
    if (test__floattixf(make_ti(0x8000010000000000LL, 0), -0x1.FFFFFCp+126))
        return 1;
    if (test__floattixf(make_ti(0x8000000000001000LL, 0), -0x1.FFFFFFFFFFFFCp+126))
        return 1;

    if (test__floattixf(make_ti(0x8000000000000000LL, 0), -0x1.000000p+127))
        return 1;
    if (test__floattixf(make_ti(0x8000000000000001LL, 0), -0x1.FFFFFFFFFFFFFFFCp+126L))
        return 1;

    if (test__floattixf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50))
        return 1;

    if (test__floattixf(0x0007FB72EA000000LL, 0x1.FEDCBA8p+50))
        return 1;
    if (test__floattixf(0x0007FB72EB000000LL, 0x1.FEDCBACp+50))
        return 1;
    if (test__floattixf(0x0007FB72EBFFFFFFLL, 0x1.FEDCBAFFFFFFCp+50))
        return 1;
    if (test__floattixf(0x0007FB72EC000000LL, 0x1.FEDCBBp+50))
        return 1;
    if (test__floattixf(0x0007FB72E8000001LL, 0x1.FEDCBA0000004p+50))
        return 1;

    if (test__floattixf(0x0007FB72E6000000LL, 0x1.FEDCB98p+50))
        return 1;
    if (test__floattixf(0x0007FB72E7000000LL, 0x1.FEDCB9Cp+50))
        return 1;
    if (test__floattixf(0x0007FB72E7FFFFFFLL, 0x1.FEDCB9FFFFFFCp+50))
        return 1;
    if (test__floattixf(0x0007FB72E4000001LL, 0x1.FEDCB90000004p+50))
        return 1;
    if (test__floattixf(0x0007FB72E4000000LL, 0x1.FEDCB9p+50))
        return 1;

    if (test__floattixf(0x023479FD0E092DC0LL, 0x1.1A3CFE870496Ep+57))
        return 1;
    if (test__floattixf(0x023479FD0E092DA1LL, 0x1.1A3CFE870496D08p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DB0LL, 0x1.1A3CFE870496D8p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DB8LL, 0x1.1A3CFE870496DCp+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DB6LL, 0x1.1A3CFE870496DBp+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DBFLL, 0x1.1A3CFE870496DF8p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DC1LL, 0x1.1A3CFE870496E08p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DC7LL, 0x1.1A3CFE870496E38p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DC8LL, 0x1.1A3CFE870496E4p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DCFLL, 0x1.1A3CFE870496E78p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DD0LL, 0x1.1A3CFE870496E8p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DD1LL, 0x1.1A3CFE870496E88p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DD8LL, 0x1.1A3CFE870496ECp+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DDFLL, 0x1.1A3CFE870496EF8p+57L))
        return 1;
    if (test__floattixf(0x023479FD0E092DE0LL, 0x1.1A3CFE870496Fp+57))
        return 1;

    if (test__floattixf(make_ti(0x023479FD0E092DC0LL, 0), 0x1.1A3CFE870496Ep+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DA1LL, 1), 0x1.1A3CFE870496D08p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DB0LL, 2), 0x1.1A3CFE870496D8p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DB8LL, 3), 0x1.1A3CFE870496DCp+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DB6LL, 4), 0x1.1A3CFE870496DBp+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DBFLL, 5), 0x1.1A3CFE870496DF8p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DC1LL, 6), 0x1.1A3CFE870496E08p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DC7LL, 7), 0x1.1A3CFE870496E38p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DC8LL, 8), 0x1.1A3CFE870496E4p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DCFLL, 9), 0x1.1A3CFE870496E78p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DD0LL, 0), 0x1.1A3CFE870496E8p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DD1LL, 11), 0x1.1A3CFE870496E88p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DD8LL, 12), 0x1.1A3CFE870496ECp+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DDFLL, 13), 0x1.1A3CFE870496EF8p+121L))
        return 1;
    if (test__floattixf(make_ti(0x023479FD0E092DE0LL, 14), 0x1.1A3CFE870496Fp+121L))
        return 1;

    if (test__floattixf(make_ti(0, 0xFFFFFFFFFFFFFFFFLL), 0x1.FFFFFFFFFFFFFFFEp+63L))
        return 1;

    if (test__floattixf(make_ti(0x0000123456789012LL, 0x3456100000000001LL),
                        0x1.2345678901234562p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x3456200000000000LL),
                        0x1.2345678901234562p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x34562FFFFFFFFFFFLL),
                        0x1.2345678901234562p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x3456300000000000LL),
                        0x1.2345678901234564p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x3456400000000000LL),
                        0x1.2345678901234564p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x34564FFFFFFFFFFFLL),
                        0x1.2345678901234564p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x3456500000000000LL),
                        0x1.2345678901234564p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x3456500000000001LL),
                        0x1.2345678901234566p+108L))
        return 1;
    if (test__floattixf(make_ti(0x0000123456789012LL, 0x34566FFFFFFFFFFFLL),
                        0x1.2345678901234566p+108L))
        return 1;
#else
    printf("skipped\n");
#endif
   return 0;
}
