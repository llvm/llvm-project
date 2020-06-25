// REQUIRES: x86-registered-target
// Test that by default -fnoopenmp-use-tls is passed to frontend.
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-DEFAULT: -cc1
// CHECK-DEFAULT-NOT: -fnoopenmp-use-tls
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp -fnoopenmp-use-tls | FileCheck --check-prefix=CHECK-NO-TLS %s

// CHECK-NO-TLS: -cc1
// CHECK-NO-TLS-SAME: -fnoopenmp-use-tls
//
// RUN: %clang %s -c -E -dM -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=1 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=0 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=100 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=50 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s

// CHECK-DEFAULT-VERSION: #define _OPENMP 201811

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=31 | FileCheck --check-prefix=CHECK-31-VERSION %s
// CHECK-31-VERSION: #define _OPENMP 201107

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=40 | FileCheck --check-prefix=CHECK-40-VERSION %s
// CHECK-40-VERSION: #define _OPENMP 201307

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 | FileCheck --check-prefix=CHECK-45-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 -fopenmp-simd | FileCheck --check-prefix=CHECK-45-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 -fopenmp-targets=x86_64-unknown-unknown -o - | FileCheck --check-prefix=CHECK-45-VERSION --check-prefix=CHECK-45-VERSION2 %s
// CHECK-45-VERSION: #define _OPENMP 201511
// CHECK-45-VERSION2: #define _OPENMP 201511

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=50 | FileCheck --check-prefix=CHECK-50-VERSION %s
// CHECK-50-VERSION: #define _OPENMP 201811

// RUN: %clang %s -c -E -dM -fopenmp-version=1 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=31 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=40 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fno-openmp-simd -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=1 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=0 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=100 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=31 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=40 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-simd -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s

// CHECK-VERSION-NOT: #define _OPENMP

