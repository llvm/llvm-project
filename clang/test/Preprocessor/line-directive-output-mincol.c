// RUN: %clang_cc1 -E -fminimize-whitespace %s 2>&1 | FileCheck %s -strict-whitespace

// CHECK:      # 6 "{{.*}}line-directive-output-mincol.c"
// CHECK-NEXT: int x;
// CHECK-NEXT: int y;
int x;
int y;
// CHECK-NEXT: # 10 "{{.*}}line-directive-output-mincol.c"
// CHECK-NEXT: int z;
int z;

