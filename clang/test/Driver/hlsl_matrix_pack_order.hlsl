// RUN: %clang_dxc -T lib_6_7  -Zpr -### %s 2>&1 | FileCheck %s  --check-prefix=CHECK-ROW-MAJOR
// CHECK-ROW-MAJOR:  -fmatrix-memory-layout=row-major

// RUN: %clang_dxc -T lib_6_7  -Zpc -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-COL-MAJOR
// CHECK-COL-MAJOR:  -fmatrix-memory-layout=column-major

// RUN: not  %clang_dxc  -Tlib_6_7 -Zpr -Zpc -fcgl -Fo - %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MISMATCH-MAJOR
// CHECK-MISMATCH-MAJOR: cannot specify /Zpr and /Zpc together
