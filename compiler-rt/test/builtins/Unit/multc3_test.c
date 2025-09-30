// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_multc3

#include <stdio.h>
#include "int_lib.h"

#if defined(CRT_HAS_128BIT) && defined(CRT_HAS_F128)

#define QUAD_PRECISION
#include "fp_lib.h"

#include <math.h>
#include <complex.h>

// Returns: the product of a + ib and c + id

COMPILER_RT_ABI Qcomplex
__multc3(fp_t __a, fp_t __b, fp_t __c, fp_t __d);

enum {zero, non_zero, inf, NaN, non_zero_nan};

int
classify(Qcomplex x)
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

int test__multc3(fp_t a, fp_t b, fp_t c, fp_t d)
{
    Qcomplex r = __multc3(a, b, c, d);
//     printf("test__multc3(%Lf, %Lf, %Lf, %Lf) = %Lf + I%Lf\n",
//             a, b, c, d, creall(r), cimagl(r));
	Qcomplex dividend;
	Qcomplex divisor;
	
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
            if (classify(r) != zero)
                return 1;
            break;
        case non_zero:
            if (classify(r) != zero)
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
    case non_zero:
        switch (classify(divisor))
        {
        case zero:
            if (classify(r) != zero)
                return 1;
            break;
        case non_zero:
            if (classify(r) != non_zero)
                return 1;
            if (r != a * c - b * d + _Complex_I*(a * d + b * c))
                return 1;
            break;
        case inf:
            if (classify(r) != inf)
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
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero:
            if (classify(r) != inf)
                return 1;
            break;
        case inf:
            if (classify(r) != inf)
                return 1;
            break;
        case NaN:
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero_nan:
            if (classify(r) != inf)
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
            if (classify(r) != NaN)
                return 1;
            break;
        case non_zero:
            if (classify(r) != NaN)
                return 1;
            break;
        case inf:
            if (classify(r) != inf)
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

fp_t x[][2] =
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

#endif

int main()
{
#if defined(CRT_HAS_128BIT) && defined(CRT_HAS_F128)
    const unsigned N = sizeof(x) / sizeof(x[0]);
    unsigned i, j;
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            if (test__multc3(x[i][0], x[i][1], x[j][0], x[j][1]))
                return 1;
        }
    }
#else
    printf("skipped\n");
#endif
    return 0;
}
