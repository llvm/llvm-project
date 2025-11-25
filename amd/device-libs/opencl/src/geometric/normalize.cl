/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, const))

ATTR float
normalize(float p)
{
    return sign(p);
}

ATTR float2
normalize(float2 p)
{
    if (all(p == (float2)0.0F))
	return p;

    float l2 = dot(p, p);

    if (l2 < FLT_MIN) {
        p *= 0x1.0p+86F;
        l2 = dot(p, p);
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-65f;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((float2)0.0F, (float2)1.0F, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR float3
normalize(float3 p)
{
    if (all(p == (float3)0.0F))
	return p;

    float l2 = dot(p, p);

    if (l2 < FLT_MIN) {
        p *= 0x1.0p+86F;
        l2 = dot(p, p);
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-66f;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((float3)0.0F, (float3)1.0F, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR float4
normalize(float4 p)
{
    if (all(p == (float4)0.0F))
	return p;

    float l2 = dot(p, p);

    if (l2 < FLT_MIN) {
        p *= 0x1.0p+86F;
        l2 = dot(p, p);
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-66f;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((float4)0.0F, (float4)1.0F, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR double
normalize(double p)
{
    return sign(p);
}

ATTR double2
normalize(double2 p)
{
    if (all(p == (double2)0.0))
	return p;

    double l2 = dot(p, p);

    if (l2 < DBL_MIN) {
        p *= 0x1.0p+563;
        l2 = dot(p, p);
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-513;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((double2)0.0, (double2)1.0, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR double3
normalize(double3 p)
{
    if (all(p == (double3)0.0))
	return p;

    double l2 = dot(p, p);

    if (l2 < DBL_MIN) {
        p *= 0x1.0p+563;
        l2 = dot(p, p);
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-514;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((double3)0.0, (double3)1.0, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR double4
normalize(double4 p)
{
    if (all(p == (double4)0.0))
	return p;

    double l2 = dot(p, p);

    if (l2 < DBL_MIN) {
        p *= 0x1.0p+563;
        l2 = dot(p, p);
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-514;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((double4)0.0, (double4)1.0, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR half
normalize(half p)
{
    return sign(p);
}

ATTR half2
normalize(half2 p)
{
    if (all(p == (half2)0.0))
	return p;

    half l2 = dot(p, p);

    if (l2 < HALF_MIN) {
        p = p * 0x1.0p+10h * 0x1.0p+7h;
        l2 = dot(p, p);
    } else if (l2 == (half)INFINITY) {
        p *= 0x1.0p-9h;
        l2 = dot(p, p);
        if (l2 == (half)INFINITY) {
            p = copysign(select((half2)0.0, (half2)1.0, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR half3
normalize(half3 p)
{
    if (all(p == (half3)0.0))
	return p;

    half l2 = dot(p, p);

    if (l2 < HALF_MIN) {
        p = p * 0x1.0p+10h * 0x1.0p+7h;
        l2 = dot(p, p);
    } else if (l2 == (half)INFINITY) {
        p *= 0x1.0p-10h;
        l2 = dot(p, p);
        if (l2 == (half)INFINITY) {
            p = copysign(select((half3)0.0, (half3)1.0, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

ATTR half4
normalize(half4 p)
{
    if (all(p == (half4)0.0))
	return p;

    half l2 = dot(p, p);

    if (l2 < HALF_MIN) {
        p = p * 0x1.0p+10h * 0x1.0p+7h;
        l2 = dot(p, p);
    } else if (l2 == (half)INFINITY) {
        p *= 0x1.0p-10h;
        l2 = dot(p, p);
        if (l2 == INFINITY) {
            p = copysign(select((half4)0.0, (half4)1.0, isinf(p)), p);
            l2 = dot(p, p);
        }
    }
    return p * rsqrt(l2);
}

