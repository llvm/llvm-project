/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, always_inline, const))

#define GEN(T) \
ATTR T \
dot(T p0, T p1) \
{ \
    return p0 * p1; \
} \
ATTR T \
dot(T##2 p0, T##2 p1) \
{ \
    return mad(p0.y, p1.y, p0.x*p1.x); \
} \
ATTR T \
dot(T##3 p0, T##3 p1) \
{ \
    return mad(p0.z, p1.z, mad(p0.y, p1.y, p0.x*p1.x)); \
} \
ATTR T \
dot(T##4 p0, T##4 p1) \
{ \
    return mad(p0.w, p1.w, mad(p0.z, p1.z, mad(p0.y, p1.y, p0.x*p1.x))); \
}

GEN(float)
GEN(double)
GEN(half)

