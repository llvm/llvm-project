// RUN: %clang_builtins %s %librt -lm -o %t && %run %t
//===-- divtc3_test.c - Test __divtc3 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __divtc3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "int_lib.h"
#include <math.h>
#include <complex.h>

// REQUIRES: c99-complex

// Returns: the quotient of (a + ib) / (c + id)

COMPILER_RT_ABI long double _Complex
__divtc3(long double __a, long double __b, long double __c, long double __d);

enum {zero, non_zero, inf, NaN, non_zero_nan};

int
classify(long double _Complex x)
{
    if (x == 0)
        return zero;
    if (isinf(creall(x)) || isinf(cimagl(x)))
        return inf;
    if (isnan(creall(x)) && isnan(cimagl(x)))
        return NaN;
    if (isnan(creall(x)))
    {
        if (cimagl(x) == 0)
            return NaN;
        return non_zero_nan;
    }
    if (isnan(cimagl(x)))
    {
        if (creall(x) == 0)
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

int test__divtc3(long double a, long double b, long double c, long double d)
{
    long double _Complex r = __divtc3(a, b, c, d);
//      printf("test__divtc3(%Lf, %Lf, %Lf, %Lf) = %Lf + I%Lf\n",
//              a, b, c, d, creall(r), cimagl(r));
	
	long double _Complex dividend;
	long double _Complex divisor;
	
	__real__ dividend = a;
	__imag__ dividend = b;
	__real__ divisor = c;
	__imag__ divisor = d;
	
    switch (classify(dividend))
    {
    case zero:
        switch (classify(divisor))
        {
        case zero:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero:
            if (classify(r) != zero)
                return 1;
            break;
        case inf:
            if (classify(r) != zero)
                return 1;
            break;
        case NaN:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero_nan:
            if (classify(r) != NaN)
                return 1;
            break;
        }
        break;
    case non_zero:
        switch (classify(divisor))
        {
        case zero:
            if (classify(r) != inf)
                return 1;
            break;
        case non_zero:
            if (classify(r) != non_zero)
                return 1;
            {
            long double _Complex z = (a * c + b * d) / (c * c + d * d)
                                   + (b * c - a * d) / (c * c + d * d) * _Complex_I;
            if (cabsl((r - z)/r) > 1.e-6)
                return 1;
            }
            break;
        case inf:
            if (classify(r) != zero)
                return 1;
            break;
        case NaN:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero_nan:
            if (classify(r) != NaN)
                return 1;
            break;
        }
        break;
    case inf:
        switch (classify(divisor))
        {
        case zero:
            if (classify(r) != inf)
                return 1;
            break;
        case non_zero:
            if (classify(r) != inf)
                return 1;
            break;
        case inf:
            if (classify(r) != NaN)
                return 1;
            break;
        case NaN:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero_nan:
            if (classify(r) != NaN)
                return 1;
            break;
        }
        break;
    case NaN:
        switch (classify(divisor))
        {
        case zero:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero:
            if (classify(r) != NaN)
                return 1;
            break;
        case inf:
            if (classify(r) != NaN)
                return 1;
            break;
        case NaN:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero_nan:
            if (classify(r) != NaN)
                return 1;
            break;
        }
        break;
    case non_zero_nan:
        switch (classify(divisor))
        {
        case zero:
            if (classify(r) != inf)
                return 1;
            break;
        case non_zero:
            if (classify(r) != NaN)
                return 1;
            break;
        case inf:
            if (classify(r) != NaN)
                return 1;
            break;
        case NaN:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero_nan:
            if (classify(r) != NaN)
                return 1;
            break;
        }
        break;
    }
    
    return 0;
}

long double x[][2] =
{
    { 1.e-6,  1.e-6},
    {-1.e-6,  1.e-6},
    {-1.e-6, -1.e-6},
    { 1.e-6, -1.e-6},

    { 1.e+6,  1.e-6},
    {-1.e+6,  1.e-6},
    {-1.e+6, -1.e-6},
    { 1.e+6, -1.e-6},

    { 1.e-6,  1.e+6},
    {-1.e-6,  1.e+6},
    {-1.e-6, -1.e+6},
    { 1.e-6, -1.e+6},

    { 1.e+6,  1.e+6},
    {-1.e+6,  1.e+6},
    {-1.e+6, -1.e+6},
    { 1.e+6, -1.e+6},

    {NAN, NAN},
    {-INFINITY, NAN},
    {-2, NAN},
    {-1, NAN},
    {-0.5, NAN},
    {-0., NAN},
    {+0., NAN},
    {0.5, NAN},
    {1, NAN},
    {2, NAN},
    {INFINITY, NAN},

    {NAN, -INFINITY},
    {-INFINITY, -INFINITY},
    {-2, -INFINITY},
    {-1, -INFINITY},
    {-0.5, -INFINITY},
    {-0., -INFINITY},
    {+0., -INFINITY},
    {0.5, -INFINITY},
    {1, -INFINITY},
    {2, -INFINITY},
    {INFINITY, -INFINITY},

    {NAN, -2},
    {-INFINITY, -2},
    {-2, -2},
    {-1, -2},
    {-0.5, -2},
    {-0., -2},
    {+0., -2},
    {0.5, -2},
    {1, -2},
    {2, -2},
    {INFINITY, -2},

    {NAN, -1},
    {-INFINITY, -1},
    {-2, -1},
    {-1, -1},
    {-0.5, -1},
    {-0., -1},
    {+0., -1},
    {0.5, -1},
    {1, -1},
    {2, -1},
    {INFINITY, -1},

    {NAN, -0.5},
    {-INFINITY, -0.5},
    {-2, -0.5},
    {-1, -0.5},
    {-0.5, -0.5},
    {-0., -0.5},
    {+0., -0.5},
    {0.5, -0.5},
    {1, -0.5},
    {2, -0.5},
    {INFINITY, -0.5},

    {NAN, -0.},
    {-INFINITY, -0.},
    {-2, -0.},
    {-1, -0.},
    {-0.5, -0.},
    {-0., -0.},
    {+0., -0.},
    {0.5, -0.},
    {1, -0.},
    {2, -0.},
    {INFINITY, -0.},

    {NAN, 0.},
    {-INFINITY, 0.},
    {-2, 0.},
    {-1, 0.},
    {-0.5, 0.},
    {-0., 0.},
    {+0., 0.},
    {0.5, 0.},
    {1, 0.},
    {2, 0.},
    {INFINITY, 0.},

    {NAN, 0.5},
    {-INFINITY, 0.5},
    {-2, 0.5},
    {-1, 0.5},
    {-0.5, 0.5},
    {-0., 0.5},
    {+0., 0.5},
    {0.5, 0.5},
    {1, 0.5},
    {2, 0.5},
    {INFINITY, 0.5},

    {NAN, 1},
    {-INFINITY, 1},
    {-2, 1},
    {-1, 1},
    {-0.5, 1},
    {-0., 1},
    {+0., 1},
    {0.5, 1},
    {1, 1},
    {2, 1},
    {INFINITY, 1},

    {NAN, 2},
    {-INFINITY, 2},
    {-2, 2},
    {-1, 2},
    {-0.5, 2},
    {-0., 2},
    {+0., 2},
    {0.5, 2},
    {1, 2},
    {2, 2},
    {INFINITY, 2},

    {NAN, INFINITY},
    {-INFINITY, INFINITY},
    {-2, INFINITY},
    {-1, INFINITY},
    {-0.5, INFINITY},
    {-0., INFINITY},
    {+0., INFINITY},
    {0.5, INFINITY},
    {1, INFINITY},
    {2, INFINITY},
    {INFINITY, INFINITY}

};

int main()
{
    const unsigned N = sizeof(x) / sizeof(x[0]);
    unsigned i, j;
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            if (test__divtc3(x[i][0], x[i][1], x[j][0], x[j][1]))
                return 1;
        }
    }

//	printf("No errors found.\n");
    return 0;
}
