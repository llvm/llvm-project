/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define CATTR __attribute__((overloadable, always_inline, const))

// XXX assumes wavefront size is 64

CATTR uint
get_sub_group_size(void)
{
    uint wgs = mul24((uint)get_local_size(2), mul24((uint)get_local_size(1), (uint)get_local_size(0)));
    uint lid = (uint)get_local_linear_id();
    return min(64U, wgs - (lid & ~63U));
}

CATTR uint
get_max_sub_group_size(void)
{
    uint wgs = mul24((uint)get_enqueued_local_size(2), mul24((uint)get_enqueued_local_size(1), (uint)get_enqueued_local_size(0)));
    return min(64U, wgs);
}

CATTR uint
get_num_sub_groups(void)
{
    uint wgs = mul24((uint)get_local_size(2), mul24((uint)get_local_size(1), (uint)get_local_size(0)));
    return (wgs + 63U) >> 6U;
}

CATTR uint
get_enqueued_num_sub_groups(void)
{
    uint wgs = mul24((uint)get_enqueued_local_size(2), mul24((uint)get_enqueued_local_size(1), (uint)get_enqueued_local_size(0)));
    return (wgs + 63U) >> 6U;
}

CATTR uint
get_sub_group_id(void)
{
    return (uint)get_local_linear_id() >> 6U;
}

CATTR uint
get_sub_group_local_id(void)
{
    return __ockl_activelane_u32();
}

