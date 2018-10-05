/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, const))

#define GEN(T) \
ATTR T##3 \
cross(T##3 p0, T##3 p1) \
{ \
    return (T##3)(mad(p0.y, p1.z, -p0.z*p1.y), \
                  mad(p0.z, p1.x, -p0.x*p1.z), \
                  mad(p0.x, p1.y, -p0.y*p1.x)); \
} \
 \
ATTR T##4 \
cross(T##4 p0, T##4 p1) \
{ \
    return (T##4)(mad(p0.y, p1.z, -p0.z*p1.y), \
                  mad(p0.z, p1.x, -p0.x*p1.z), \
                  mad(p0.x, p1.y, -p0.y*p1.x), \
                  (T)0); \
}

GEN(float)
GEN(double)
GEN(half)
