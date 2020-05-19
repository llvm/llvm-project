/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ATTR __attribute__((overloadable, const))

ATTR size_t
get_global_offset(uint dim)
{
    return __ockl_get_global_offset(dim);
}

ATTR size_t
get_global_id(uint dim)
{
    return __ockl_get_global_id(dim);
}

ATTR size_t
get_local_id(uint dim)
{
    return __ockl_get_local_id(dim);
}

ATTR size_t
get_group_id(uint dim)
{
    return __ockl_get_group_id(dim);
}

ATTR size_t
get_global_size(uint dim)
{
    return __ockl_get_global_size(dim);
}

ATTR size_t
get_local_size(uint dim)
{
    return __ockl_get_local_size(dim);
}

ATTR size_t
get_num_groups(uint dim)
{
    return __ockl_get_num_groups(dim);
}

ATTR uint
get_work_dim(void)
{
    return __ockl_get_work_dim();
}

ATTR size_t
get_enqueued_local_size(uint dim)
{
    return __ockl_get_enqueued_local_size(dim);
}

ATTR size_t
get_global_linear_id(void)
{
    return __ockl_get_global_linear_id();
}

ATTR size_t
get_local_linear_id(void)
{
    return __ockl_get_local_linear_id();
}

