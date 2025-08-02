// RUN: %clang_cc1 -triple aarch64 -ast-dump -ast-dump-filter foo %s \
// RUN: | FileCheck --strict-whitespace %s

// CHECK: foo1 'void () __attribute__((device_kernel))'{{$}}
void foo1() __attribute__((device_kernel));

// CHECK: foo2 'void () __attribute__((aarch64_vector_pcs))'{{$}}
void foo2()  __attribute__((aarch64_vector_pcs));

// CHECK: foo3 'void () __attribute__((aarch64_sve_pcs))'{{$}}
void foo3()  __attribute__((aarch64_sve_pcs));
