// REQUIRES: x86-registered-target
// REQUIRES: clang-target-64-bits
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 -fopenmp-targets=x86_64-unknown-unknown -o - | FileCheck --check-prefix=CHECK-45-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 -fopenmp-targets=x86_64-unknown-unknown -fopenmp-simd | FileCheck --check-prefix=CHECK-45-VERSION2 %s
// CHECK-45-VERSION: #define _OPENMP 201511
// CHECK-45-VERSION2: #define _OPENMP 201511
