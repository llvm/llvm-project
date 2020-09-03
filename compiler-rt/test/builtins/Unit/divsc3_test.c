// RUN: %clang_builtins %s %librt -lm -o %t && %run %t
// REQUIRES: librt_has_divsc3
// REQUIRES: c99-complex

#include "int_lib.h"
#include <math.h>
#include <complex.h>
#include <stdio.h>


// Returns: the quotient of (a + ib) / (c + id)

COMPILER_RT_ABI float _Complex
__divsc3(float __a, float __b, float __c, float __d);

enum {zero, non_zero, inf, NaN, non_zero_nan};

int
classify(float _Complex x)
{
    if (x == 0)
        return zero;
    if (isinf(crealf(x)) || isinf(cimagf(x)))
        return inf;
    if (isnan(crealf(x)) && isnan(cimagf(x)))
        return NaN;
    if (isnan(crealf(x)))
    {
        if (cimagf(x) == 0)
            return NaN;
        return non_zero_nan;
    }
    if (isnan(cimagf(x)))
    {
        if (crealf(x) == 0)
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

int test__divsc3(float a, float b, float c, float d)
{
    float _Complex r = __divsc3(a, b, c, d);
//     printf("test__divsc3(%f, %f, %f, %f) = %f + I%f\n",
//             a, b, c, d, crealf(r), cimagf(r));
	float _Complex dividend;
	float _Complex divisor;
	
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
            float _Complex z = (a * c + b * d) / (c * c + d * d)
                             + (b * c - a * d) / (c * c + d * d) * _Complex_I;
            if (cabsf((r-z)/r) > 1.e-6)
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

float x[][2] =
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
            if (test__divsc3(x[i][0], x[i][1], x[j][0], x[j][1]))
                return 1;
        }
    }

    return 0;
}
