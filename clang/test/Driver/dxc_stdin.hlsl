// RUN: %clang_dxc -Tlib_6_7 - -### 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK-SAME: "-x" "hlsl" "-"
