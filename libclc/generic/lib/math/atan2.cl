/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "math.h"
#include "../clcmacro.h"

#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF float atan2(float y, float x)
{
    const float pi = 0x1.921fb6p+1f;
    const float piby2 = 0x1.921fb6p+0f;
    const float piby4 = 0x1.921fb6p-1f;
    const float threepiby4 = 0x1.2d97c8p+1f;

    float ax = fabs(x);
    float ay = fabs(y);
    float v = min(ax, ay);
    float u = max(ax, ay);

    // Scale since u could be large, as in "regular" divide
    float s = u > 0x1.0p+96f ? 0x1.0p-32f : 1.0f;
    float vbyu = s * MATH_DIVIDE(v, s*u);

    float vbyu2 = vbyu * vbyu;

#define USE_2_2_APPROXIMATION
#if defined USE_2_2_APPROXIMATION
    float p = mad(vbyu2, mad(vbyu2, -0x1.7e1f78p-9f, -0x1.7d1b98p-3f), -0x1.5554d0p-2f) * vbyu2 * vbyu;
    float q = mad(vbyu2, mad(vbyu2, 0x1.1a714cp-2f, 0x1.287c56p+0f), 1.0f);
#else
    float p = mad(vbyu2, mad(vbyu2, -0x1.55cd22p-5f, -0x1.26cf76p-2f), -0x1.55554ep-2f) * vbyu2 * vbyu;
    float q = mad(vbyu2, mad(vbyu2, mad(vbyu2, 0x1.9f1304p-5f, 0x1.2656fap-1f), 0x1.76b4b8p+0f), 1.0f);
#endif

    // Octant 0 result
    float a = mad(p, MATH_RECIP(q), vbyu);

    // Fix up 3 other octants
    float at = piby2 - a;
    a = ay > ax ? at : a;
    at = pi - a;
    a = x < 0.0F ? at : a;

    // y == 0 => 0 for x >= 0, pi for x < 0
    at = as_int(x) < 0 ? pi : 0.0f;
    a = y == 0.0f ? at : a;

    // if (!FINITE_ONLY()) {
        // x and y are +- Inf
        at = x > 0.0f ? piby4 : threepiby4;
        a = ax == INFINITY & ay == INFINITY ? at : a;

	// x or y is NaN
	a = isnan(x) | isnan(y) ? as_float(QNANBITPATT_SP32) : a;
    // }

    // Fixup sign and return
    return copysign(a, y);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, atan2, float, float);
