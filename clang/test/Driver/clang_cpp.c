// Verify that -include isn't included twice with -save-temps.
// RUN: %clang -S -o - %s -include %t.h -save-temps -### 2> %t.log
// RUN: FileCheck %s < %t.log
// CHECK: "-include
// CHECK-NOT: "-include
