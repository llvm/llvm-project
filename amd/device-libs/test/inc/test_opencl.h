/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef TEST_OPENCL_H
#define TEST_OPENCL_H

#define \
  TEST_KERNEL_FUNC_NO_ARGS(func, type_res) \
    kernel void test_##func(global type_res* out) \
    { \
      out[0] = func(); \
    }


#define \
  TEST_KERNEL_FUNC_DIM(func, type_res, dim) \
    kernel void test_##func##_##dim(global type_res* out) \
    { \
      out[0] = func(dim); \
    }

#define \
  TEST_KERNEL_FUNC_DIMS(func, type_res) \
    TEST_KERNEL_FUNC_DIM(func, type_res, 0) \
    TEST_KERNEL_FUNC_DIM(func, type_res, 1) \
    TEST_KERNEL_FUNC_DIM(func, type_res, 2) \

#define \
  TEST_KERNEL_FUNC(func, type_res, type_arg) \
    kernel void test_##func##_##type_res##_##type_arg(global type_res* out, \
      global type_arg* in) \
    { \
      out[0] = func(in[0]); \
    }

#define \
  VEC_REPEAT2(NAME, func, arg1, arg2) \
    NAME(func, arg1, arg2) \
    NAME(func, arg1##2, arg2##2) \
    NAME(func, arg1##3, arg2##3) \
    NAME(func, arg1##4, arg2##4) \
    NAME(func, arg1##8, arg2##8) \
    NAME(func, arg1##16, arg2##16)

#define \
  VEC_REPEAT2_1(NAME, func, arg1, arg2) \
    NAME(func, arg1##2, arg2) \
    NAME(func, arg1##3, arg2) \
    NAME(func, arg1##4, arg2) \
    NAME(func, arg1##8, arg2) \
    NAME(func, arg1##16, arg2)

#define \
  VEC_REPEAT3(NAME, func, arg1, arg2, arg3) \
    NAME(func, arg1, arg2, arg3) \
    NAME(func, arg1##2, arg2##2, arg3##2) \
    NAME(func, arg1##3, arg2##3, arg3##3) \
    NAME(func, arg1##4, arg2##4, arg3##4) \
    NAME(func, arg1##8, arg2##8, arg3##8) \
    NAME(func, arg1##16, arg2##16, arg3##16)

#define \
  VEC_REPEAT3_1(NAME, func, arg1, arg2, arg3) \
    NAME(func, arg1##2, arg2##2, arg3) \
    NAME(func, arg1##3, arg2##3, arg3) \
    NAME(func, arg1##4, arg2##4, arg3) \
    NAME(func, arg1##8, arg2##8, arg3) \
    NAME(func, arg1##16, arg2##16, arg3)

#define \
  VEC_REPEAT3_1r(NAME, func, arg1, arg2, arg3) \
    NAME(func, arg1##2, arg2, arg3##2) \
    NAME(func, arg1##3, arg2, arg3##3) \
    NAME(func, arg1##4, arg2, arg3##4) \
    NAME(func, arg1##8, arg2, arg3##8) \
    NAME(func, arg1##16, arg2, arg3##16)

#define \
  VEC_REPEAT4(NAME, func, arg1, arg2, arg3, arg4) \
    NAME(func, arg1, arg2, arg3, arg4) \
    NAME(func, arg1##2, arg2##2, arg3##2, arg4##2) \
    NAME(func, arg1##3, arg2##3, arg3##3, arg4##3) \
    NAME(func, arg1##4, arg2##4, arg3##4, arg4##4) \
    NAME(func, arg1##8, arg2##8, arg3##8, arg4##8) \
    NAME(func, arg1##16, arg2##16, arg3##16, arg4##16)

#define \
  VEC_REPEAT4_1(NAME, func, arg1, arg2, arg3, arg4) \
    NAME(func, arg1##2, arg2##2, arg3##2, arg4) \
    NAME(func, arg1##3, arg2##3, arg3##3, arg4) \
    NAME(func, arg1##4, arg2##4, arg3##4, arg4) \
    NAME(func, arg1##8, arg2##8, arg3##8, arg4) \
    NAME(func, arg1##16, arg2##16, arg3##16, arg4)

#define \
  VEC_REPEAT4_1r(NAME, func, arg1, arg2, arg3, arg4) \
    NAME(func, arg1##2, arg2, arg3##2, arg4##2) \
    NAME(func, arg1##3, arg2, arg3##3, arg4##3) \
    NAME(func, arg1##4, arg2, arg3##4, arg4##4) \
    NAME(func, arg1##8, arg2, arg3##8, arg4##8) \
    NAME(func, arg1##16, arg2, arg3##16, arg4##16)

#define \
  VEC_REPEAT4_2(NAME, func, arg1, arg2, arg3, arg4) \
    NAME(func, arg1##2, arg2##2, arg3, arg4) \
    NAME(func, arg1##3, arg2##3, arg3, arg4) \
    NAME(func, arg1##4, arg2##4, arg3, arg4) \
    NAME(func, arg1##8, arg2##8, arg3, arg4) \
    NAME(func, arg1##16, arg2##16, arg3, arg4)

#define \
  VEC_REPEAT4_2r(NAME, func, arg1, arg2, arg3, arg4) \
    NAME(func, arg1##2, arg2, arg3, arg4##2) \
    NAME(func, arg1##3, arg2, arg3, arg4##3) \
    NAME(func, arg1##4, arg2, arg3, arg4##4) \
    NAME(func, arg1##8, arg2, arg3, arg4##8) \
    NAME(func, arg1##16, arg2, arg3, arg4##16)

#define \
  TEST_KERNEL_FUNC_VEC(func, type_res, type_arg) \
    VEC_REPEAT2(TEST_KERNEL_FUNC, func, type_res, type_arg)

#define \
  TEST_KERNEL_FUNC2(func, type_res, type_arg1, type_arg2) \
    kernel void test_##func##_##type_res##_##type_arg1##_##type_arg2(global type_res* out, \
      global type_arg1* in1, global type_arg2* in2) \
    { \
      out[0] = func(in1[0], in2[0]); \
    }

#define \
  TEST_KERNEL_FUNC2_VEC(func, type_res, type_arg1, type_arg2) \
    VEC_REPEAT3(TEST_KERNEL_FUNC2, func, type_res, type_arg1, type_arg2)

#define \
  TEST_KERNEL_FUNC2p(func, type_res, type_res2, type_arg1) \
    kernel void test_##func##_##type_res##_##type_res2##_##type_arg1(global type_res* out, \
      global type_res2* out2, global type_arg1* in1) \
    { \
      type_res2 v; \
      out[0] = func(in1[0], &v); \
      out2[0] = v; \
    }

#define \
  TEST_KERNEL_FUNC2p_VEC(func, type_res, type_res2, type_arg1) \
    VEC_REPEAT3(TEST_KERNEL_FUNC2p, func, type_res, type_res2, type_arg1)

#define \
  TEST_KERNEL_FUNC3(func, type_res, type_arg1, type_arg2, type_arg3) \
    kernel void test_##func##_##type_res##_##type_arg1##_##type_arg2##_##type_arg3(global type_res* out, \
      global type_arg1* in1, global type_arg2* in2, global type_arg3* in3) \
    { \
      out[0] = func(in1[0], in2[0], in3[0]); \
    }

#define \
  TEST_KERNEL_FUNC3_VEC(func, type_res, type_arg1, type_arg2, type_arg3) \
    VEC_REPEAT4(TEST_KERNEL_FUNC3, func, type_res, type_arg1, type_arg2, type_arg3)

#define \
  TEST_KERNEL_FUNC3p(func, type_res, type_res2, type_arg1, type_arg2) \
    kernel void test_##func##_##type_res##_##type_res2##_##type_arg1##_##type_arg2(global type_res* out, \
      global type_res2* out2, global type_arg1* in1, global type_arg2* in2) \
    { \
      type_res2 v; \
      out[0] = func(in1[0], in2[0], &v); \
      out2[0] = v; \
    }

#define \
  TEST_KERNEL_FUNC3p_VEC(func, type_res, type_res2, type_arg1, type_arg2) \
    VEC_REPEAT4(TEST_KERNEL_FUNC3p, func, type_res, type_res2, type_arg1, type_arg2)

#endif /* TEST_OPENCL_H */
