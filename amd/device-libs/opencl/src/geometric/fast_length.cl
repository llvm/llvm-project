/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((overloadable, const))

ATTR float
fast_length(float p)
{
    return fabs(p);
}

ATTR float
fast_length(float2 p)
{
    return half_sqrt(dot(p, p));
}

ATTR float
fast_length(float3 p)
{
    return half_sqrt(dot(p, p));
}

ATTR float
fast_length(float4 p)
{
    return half_sqrt(dot(p, p));
}

