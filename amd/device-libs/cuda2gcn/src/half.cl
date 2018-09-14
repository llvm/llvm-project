/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((const))

//-------- T __nv_float2half_rn
half __nv_float2half_rn(float x)
{
    return (half)x;
}

//-------- T __nv_half2float
float __nv_half2float(half x)
{
    return (float)x;
}

