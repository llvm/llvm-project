// RUN: %clang_cc1 -x cuda -std=c++11 -DCUDA %s
// RUN: %clang_cc1 -x cl -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=cl -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=cl1.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=cl1.1 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=cl1.2 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=cl2.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=cl3.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=clc++ -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=clc++1.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=clc++2021 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CL -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CL1.1 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CL3.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CLC++ -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CLC++1.0 -DOPENCL %s
// RUN: %clang_cc1 -x cl -cl-std=CLC++2021 -DOPENCL %s
// RUN: not %clang_cc1 -x cl -std=c99 -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-C99 %s
// RUN: not %clang_cc1 -x cl -cl-std=invalid -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s
// CHECK-C99: error: invalid argument '-std=c99' not allowed with 'OpenCL'
// CHECK-INVALID: error: invalid value 'invalid' in '-cl-std=invalid'

#if defined(CUDA)
  __attribute__((device)) void f_device();
#elif defined(OPENCL)
  kernel void func(void);
#endif
