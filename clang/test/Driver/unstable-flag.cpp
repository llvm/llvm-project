// RUN: %clang -funstable -### %s 2>&1 | FileCheck %s

// CHECK: -funstable
// CHECK: -fcoroutines-ts
// CHECK: -fmodules-ts
