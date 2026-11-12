// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatundisf

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

// Returns: convert a to a float, rounding toward even.

// Assumption: float is a IEEE 32 bit floating point type 
//             du_int is a 64 bit integral type

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI float __floatundisf(du_int a);

int test__floatundisf(du_int a, float expected)
{
    float x = __floatundisf(a);
    if (x != expected)
        printf("error in __floatundisf(%llX) = %a, expected %a\n",
               a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(du_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(du_int)*CHAR_BIT == 64] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

int main()
{
    if (test__floatundisf(0, 0.0F))
        return 1;

    if (test__floatundisf(1, 1.0F))
        return 1;
    if (test__floatundisf(2, 2.0F))
        return 1;

    if (test__floatundisf(0x7FFFFF8000000000ULL, 0x1.FFFFFEp+62F))
        return 1;
    if (test__floatundisf(0x7FFFFF0000000000ULL, 0x1.FFFFFCp+62F))
        return 1;

    if (test__floatundisf(0x8000008000000000ULL, 0x1p+63F))
        return 1;
    if (test__floatundisf(0x8000010000000000LL, 0x1.000002p+63F))
        return 1;

    if (test__floatundisf(0x8000000000000000LL, 0x1p+63F))
        return 1;
    if (test__floatundisf(0x8000000000000001LL, 0x1p+63F))
        return 1;

    if (test__floatundisf(0xFFFFFFFFFFFFFFFELL, 0x1p+64F))
        return 1;
    if (test__floatundisf(0xFFFFFFFFFFFFFFFFLL, 0x1p+64F))
        return 1;

    if (test__floatundisf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floatundisf(0x0007FB72EA000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72EB000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72EBFFFFFFLL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72EC000000LL, 0x1.FEDCBCp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72E8000001LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floatundisf(0x0007FB72E6000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72E7000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72E7FFFFFFLL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72E4000001LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatundisf(0x0007FB72E4000000LL, 0x1.FEDCB8p+50F))
        return 1;

   return 0;
}
