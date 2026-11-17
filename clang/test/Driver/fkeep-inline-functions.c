// RUN: %clang -fkeep-inline-functions -c %s -### 2>&1 | FileCheck %s
// RUN: %clang -fkeep-inline-functions -fno-keep-inline-functions -c %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-NOKEEP

// CHECK: "-fkeep-inline-functions"
// CHECK-NOKEEP-NOT: "-fkeep-inline-functions"
