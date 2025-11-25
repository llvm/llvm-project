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
    return min(OCLC_WAVEFRONT_SIZE, wgs - (lid & ~(OCLC_WAVEFRONT_SIZE - 1)));
}

CATTR uint
get_max_sub_group_size(void)
{
    uint wgs = mul24((uint)get_enqueued_local_size(2), mul24((uint)get_enqueued_local_size(1), (uint)get_enqueued_local_size(0)));
    return min(OCLC_WAVEFRONT_SIZE, wgs);
}

CATTR uint
get_num_sub_groups(void)
{
    uint wgs = mul24((uint)get_local_size(2), mul24((uint)get_local_size(1), (uint)get_local_size(0)));
    return (wgs + OCLC_WAVEFRONT_SIZE - 1) >> __oclc_wavefrontsize_log2;
}

CATTR uint
get_enqueued_num_sub_groups(void)
{
    uint wgs = mul24((uint)get_enqueued_local_size(2), mul24((uint)get_enqueued_local_size(1), (uint)get_enqueued_local_size(0)));
    return (wgs + OCLC_WAVEFRONT_SIZE - 1) >> __oclc_wavefrontsize_log2;
}

CATTR uint
get_sub_group_id(void)
{

    return (uint)get_local_linear_id() >> __oclc_wavefrontsize_log2;
}

CATTR uint
get_sub_group_local_id(void)
{
    return __ockl_lane_u32();
}

