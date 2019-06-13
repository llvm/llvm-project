/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

#define CATTR __attribute__((overloadable, const))

CATTR uint
get_sub_group_size(void)
{
    uint wgs = mul24((uint)get_local_size(2), mul24((uint)get_local_size(1), (uint)get_local_size(0)));
    uint lid = (uint)get_local_linear_id();
    if (__oclc_wavefrontsize64)
        return min(64U, wgs - (lid & ~63U));
    else
        return min(32U, wgs - (lid & ~31U));
}

CATTR uint
get_max_sub_group_size(void)
{
    uint wgs = mul24((uint)get_enqueued_local_size(2), mul24((uint)get_enqueued_local_size(1), (uint)get_enqueued_local_size(0)));
    return min(__oclc_wavefrontsize64 ? 64u : 32u, wgs);
}

CATTR uint
get_num_sub_groups(void)
{
    uint wgs = mul24((uint)get_local_size(2), mul24((uint)get_local_size(1), (uint)get_local_size(0)));
    if (__oclc_wavefrontsize64)
        return (wgs + 63U) >> 6U;
    else
        return (wgs + 31U) >> 5U;
}

CATTR uint
get_enqueued_num_sub_groups(void)
{
    uint wgs = mul24((uint)get_enqueued_local_size(2), mul24((uint)get_enqueued_local_size(1), (uint)get_enqueued_local_size(0)));
    if (__oclc_wavefrontsize64)
        return (wgs + 63U) >> 6U;
    else
        return (wgs + 31U) >> 5U;
}

CATTR uint
get_sub_group_id(void)
{
    if (__oclc_wavefrontsize64)
        return (uint)get_local_linear_id() >> 6U;
    else
        return (uint)get_local_linear_id() >> 5U;
}

CATTR uint
get_sub_group_local_id(void)
{
    return __ockl_lane_u32();
}

