/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "test_opencl.h"

TEST_KERNEL_FUNC3_VEC(clamp, float, float, float, float)
VEC_REPEAT4_2(TEST_KERNEL_FUNC3, clamp, float, float, float, float)
TEST_KERNEL_FUNC_VEC(degrees, float, float)
TEST_KERNEL_FUNC2_VEC(max, float, float, float)
VEC_REPEAT3_1(TEST_KERNEL_FUNC2, max, float, float, float)
TEST_KERNEL_FUNC2_VEC(min, float, float, float)
VEC_REPEAT3_1(TEST_KERNEL_FUNC2, min, float, float, float)
TEST_KERNEL_FUNC3_VEC(mix, float, float, float, float)
VEC_REPEAT4_1(TEST_KERNEL_FUNC3, mix, float, float, float, float)
TEST_KERNEL_FUNC_VEC(radians, float, float)
TEST_KERNEL_FUNC_VEC(sign, float, float)
TEST_KERNEL_FUNC3_VEC(smoothstep, float, float, float, float)
VEC_REPEAT4_2r(TEST_KERNEL_FUNC3, smoothstep, float, float, float, float)
TEST_KERNEL_FUNC2_VEC(step, float, float, float)
VEC_REPEAT3_1r(TEST_KERNEL_FUNC2, step, float, float, float)

TEST_KERNEL_FUNC3_VEC(clamp, double, double, double, double)
VEC_REPEAT4_2(TEST_KERNEL_FUNC3, clamp, double, double, double, double)
TEST_KERNEL_FUNC_VEC(degrees, double, double)
TEST_KERNEL_FUNC2_VEC(max, double, double, double)
VEC_REPEAT3_1(TEST_KERNEL_FUNC2, max, double, double, double)
TEST_KERNEL_FUNC2_VEC(min, double, double, double)
VEC_REPEAT3_1(TEST_KERNEL_FUNC2, min, double, double, double)
TEST_KERNEL_FUNC3_VEC(mix, double, double, double, double)
VEC_REPEAT4_1(TEST_KERNEL_FUNC3, mix, double, double, double, double)
TEST_KERNEL_FUNC_VEC(radians, double, double)
TEST_KERNEL_FUNC_VEC(sign, double, double)
TEST_KERNEL_FUNC3_VEC(smoothstep, double, double, double, double)
VEC_REPEAT4_2r(TEST_KERNEL_FUNC3, smoothstep, double, double, double, double)
TEST_KERNEL_FUNC2_VEC(step, double, double, double)
VEC_REPEAT3_1r(TEST_KERNEL_FUNC2, step, double, double, double)
