/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "test_opencl.h"

TEST_KERNEL_FUNC_DIMS(get_global_offset, size_t)
TEST_KERNEL_FUNC_DIMS(get_global_id, size_t)
TEST_KERNEL_FUNC_DIMS(get_local_id, size_t)
TEST_KERNEL_FUNC_DIMS(get_group_id, size_t)
TEST_KERNEL_FUNC_DIMS(get_global_size, size_t)
TEST_KERNEL_FUNC_DIMS(get_local_size, size_t)
TEST_KERNEL_FUNC_DIMS(get_num_groups, size_t)
TEST_KERNEL_FUNC_NO_ARGS(get_work_dim, uint)
TEST_KERNEL_FUNC_DIMS(get_enqueued_local_size, size_t)
TEST_KERNEL_FUNC_NO_ARGS(get_global_linear_id, uint)
TEST_KERNEL_FUNC_NO_ARGS(get_local_linear_id, uint)
