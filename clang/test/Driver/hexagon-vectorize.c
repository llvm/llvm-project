// RUN: %clang -target hexagon -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -target hexagon -fvectorize -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-VECTOR
// RUN: %clang -target hexagon -fvectorize -fno-vectorize -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOVECTOR
// RUN: %clang -target hexagon -fvectorize -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NEEDHVX

// CHECK-DEFAULT: -hexagon-autohvx={{false|0}}
// CHECK-VECTOR-NOT: -hexagon-autohvx={{false|0}}
// CHECK-NOVECTOR: -hexagon-autohvx={{false|0}}
// CHECK-NEEDHVX: warning: auto-vectorization requires HVX, use -mhvx to enable it
