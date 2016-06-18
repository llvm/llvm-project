// RUN: %clang %s
// RUN: %clang -std=cl %s
// RUN: %clang -std=cl1.1 %s
// RUN: %clang -std=cl1.2 %s
// RUN: %clang -std=cl2.0 %s
// RUN: %clang -std=CL %s
// RUN: %clang -std=CL1.1 %s
// RUN: %clang -std=CL1.2 %s
// RUN: %clang -std=CL2.0 %s
// RUN: not %clang_cc1 -std=c99 -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-C99 %s
// RUN: not %clang_cc1 -std=invalid -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s
// CHECK-C99: error: invalid argument '-std=c99' not allowed with 'OpenCL'
// CHECK-INVALID: error: invalid value 'invalid' in '-std=invalid'

kernel void func(void);
