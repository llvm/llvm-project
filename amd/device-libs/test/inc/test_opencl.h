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

#endif /* TEST_OPENCL_H */
