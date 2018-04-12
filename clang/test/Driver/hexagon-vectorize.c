// RUN: %clang -target hexagon -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -target hexagon -fvectorize -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-VECTOR
// RUN: %clang -target hexagon -fvectorize -fno-vectorize -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOVECTOR

// CHECK-DEFAULT-NOT: hexagon-autohvx
// CHECK-VECTOR: "-mllvm" "-hexagon-autohvx"
// CHECK-NOVECTOR-NOT: hexagon-autohvx
