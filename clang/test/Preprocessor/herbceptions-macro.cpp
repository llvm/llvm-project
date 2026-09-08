// RUN: %clang -fherbceptions -dM -E -x c++ %s | FileCheck %s
// RUN: %clang -dM -E -x c++ %s | FileCheck %s --check-prefix=DISABLED

// -fherbceptions defines __HERBCEPTIONS__ so users can detect the feature.
// CHECK: #define __HERBCEPTIONS__ 1
// DISABLED-NOT: __HERBCEPTIONS__
