
#include "devenq.h"

__attribute__((always_inline, const)) uint
__get_kernel_work_group_size_impl(void *b, void *c)
{
    return (uint)CL_DEVICE_MAX_WORK_GROUP_SIZE;
}

__attribute__((always_inline, const)) uint
__get_kernel_preferred_work_group_size_multiple_impl(void *b, void *c)
{
    return 64U;
}

// 2.1 Reference card mentions
// uint get_kernel_sub_group_count_for_ndrange(ndrange_t, block);
// --> __get_kernel_sub_group_count_for_ndrange_impl(ndrange_t, void *, void *);
// uint get_kernel_max_sub_group_size_for_ndrange(ndrange_t, block);
// --> __get_kernel_max_sub_group_size_for_ndrange_impl(ndrange_t, void *, void *);
