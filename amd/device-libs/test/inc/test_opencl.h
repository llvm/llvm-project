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
  TEST_KERNEL_FUNC2(func, type_res, type_arg1, type_arg2) \
    kernel void test_##func##_##type_res##_##type_arg1##_##type_arg2(global type_res* out, \
      global type_arg1* in1, global type_arg2* in2) \
    { \
      out[0] = func(in1[0], in2[0]); \
    }

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
  TEST_KERNEL_FUNC3(func, type_res, type_arg1, type_arg2, type_arg3) \
    kernel void test_##func##_##type_res##_##type_arg1##_##type_arg2##_##type_arg3(global type_res* out, \
      global type_arg1* in1, global type_arg2* in2, global type_arg3* in3) \
    { \
      out[0] = func(in1[0], in2[0], in3[0]); \
    }

#define \
  TEST_KERNEL_FUNC3p(func, type_res, type_res2, type_arg1, type_arg2) \
    kernel void test_##func##_##type_res##_##type_res2##_##type_arg1##_##type_arg2(global type_res* out, \
      global type_res2* out2, global type_arg1* in1, global type_arg2* in2) \
    { \
      type_res2 v; \
      out[0] = func(in1[0], in2[0], &v); \
      out2[0] = v; \
    }

#endif /* TEST_OPENCL_H */
