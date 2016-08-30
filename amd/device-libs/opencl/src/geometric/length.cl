/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((always_inline, overloadable, const))

ATTR float
length(float p)
{
    return fabs(p);
}

ATTR float
length(float2 p)
{
    float l2 = dot(p, p);
    float r;

    if (l2 < FLT_MIN) {
        p *= 0x1.0p+86f;
        r = sqrt(dot(p, p)) * 0x1.0p-86f;
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-65f;
        r = sqrt(dot(p, p)) * 0x1.0p+65f;
    } else
        r = sqrt(l2);

    return r;
}

ATTR float
length(float3 p)
{
    float l2 = dot(p, p);
    float r;

    if (l2 < FLT_MIN) {
        p *= 0x1.0p+86f;
        r = sqrt(dot(p, p)) * 0x1.0p-86f;
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-66f;
        r = sqrt(dot(p, p)) * 0x1.0p+66f;
    } else
        r = sqrt(l2);

    return r;
}

ATTR float
length(float4 p)
{
    float l2 = dot(p, p);
    float r;

    if (l2 < FLT_MIN) {
        p *= 0x1.0p+86f;
        r = sqrt(dot(p, p)) * 0x1.0p-86f;
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-66f;
        r = sqrt(dot(p, p)) * 0x1.0p+66f;
    } else
        r = sqrt(l2);

    return r;
}

ATTR double
length(double p)
{
    return fabs(p);
}

ATTR double
length(double2 p)
{
    double l2 = dot(p, p);
    double r;

    if (l2 < DBL_MIN) {
        p *= 0x1.0p+563;
        r = sqrt(dot(p, p)) * 0x1.0p-563;
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-513;
        r = sqrt(dot(p, p)) * 0x1.0p+513;
    } else
        r = sqrt(l2);

    return r;
}

ATTR double
length(double3 p)
{
    double l2 = dot(p, p);
    double r;

    if (l2 < DBL_MIN) {
        p *= 0x1.0p+563;
        r = sqrt(dot(p, p)) * 0x1.0p-563;
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-514;
        r = sqrt(dot(p, p)) * 0x1.0p+514;
    } else
        r = sqrt(l2);

    return r;
}

ATTR double
length(double4 p)
{
    double l2 = dot(p, p);
    double r;

    if (l2 < DBL_MIN) {
        p *= 0x1.0p+563;
        r = sqrt(dot(p, p)) * 0x1.0p-563;
    } else if (l2 == INFINITY) {
        p *= 0x1.0p-514;
        r = sqrt(dot(p, p)) * 0x1.0p+514;
    } else
        r = sqrt(l2);

    return r;
}

ATTR half
length(half p)
{
    return fabs(p);
}

ATTR half
length(half2 p)
{
    half l2 = dot(p, p);
    half r;

    if (l2 < HALF_MIN) {
        p = p * 0x1.0p+10h * 0x1.0p+7h;
        r = sqrt(dot(p, p)) * 0x1.0p-17h;
    } else if (l2 == (half)INFINITY) {
        p *= 0x1.0p-9h;
        r = sqrt(dot(p, p)) * 0x1.0p+9h;
    } else
        r = sqrt(l2);

    return r;
}

ATTR half
length(half3 p)
{
    half l2 = dot(p, p);
    half r;

    if (l2 < HALF_MIN) {
        p = p * 0x1.0p+10h * 0x1.0p+7h;
        r = sqrt(dot(p, p)) * 0x1.0p-17h;
    } else if (l2 == (half)INFINITY) {
        p *= 0x1.0p-10h;
        r = sqrt(dot(p, p)) * 0x1.0p+10h;
    } else
        r = sqrt(l2);

    return r;
}

ATTR half
length(half4 p)
{
    half l2 = dot(p, p);
    half r;

    if (l2 < HALF_MIN) {
        p = p * 0x1.0p+10h * 0x1.0p+7h;
        r =  sqrt(dot(p, p)) * 0x1.0p-17h;
    } else if (l2 == (half)INFINITY) {
        p *= 0x1.0p-10h;
        r =  sqrt(dot(p, p)) * 0x1.0p+10h;
    } else
        r = sqrt(l2);

    return r;
}

