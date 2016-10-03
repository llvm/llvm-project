/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, always_inline))

ATTR int
sub_group_broadcast(int x, uint i)
{
    return (int)OCKL_MANGLE_U32(wfbcast)((uint)x, i);
}

ATTR uint
sub_group_broadcast(uint x, uint i)
{
    return OCKL_MANGLE_U32(wfbcast)(x, i);
}

ATTR long
sub_group_broadcast(long x, uint i)
{
    return (long)OCKL_MANGLE_U64(wfbcast)((ulong)x, i);
}

ATTR ulong
sub_group_broadcast(ulong x, uint i)
{
    return OCKL_MANGLE_U64(wfbcast)(x, i);
}

ATTR float
sub_group_broadcast(float x, uint i)
{
    return as_float(OCKL_MANGLE_U32(wfbcast)(as_uint(x), i));
}

ATTR double
sub_group_broadcast(double x, uint i)
{
    return as_double(OCKL_MANGLE_U64(wfbcast)(as_ulong(x), i));
}

ATTR half
sub_group_broadcast(half x, uint i)
{
    return as_half((ushort)OCKL_MANGLE_U32(wfbcast)((uint)as_ushort(x), i));
}

