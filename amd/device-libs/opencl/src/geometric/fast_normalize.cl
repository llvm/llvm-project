/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((overloadable, const))

#define GEN(N) \
ATTR float##N \
fast_normalize(float##N p) \
{ \
    float l2 = dot(p, p); \
    float##N n = p * half_rsqrt(l2); \
    return l2 == 0.0f ? p : n; \
}

GEN(4)
GEN(3)
GEN(2)

ATTR float
fast_normalize(float p)
{
    return sign(p);
}

