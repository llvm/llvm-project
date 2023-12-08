// RUN: not clang-query -c foo -c bar %s -- | FileCheck %s
// RUN: not clang-query -f %S/Inputs/foo.script %s -- | FileCheck %s
// RUN: not clang-query -f %S/Inputs/nonexistent.script %s -- 2>&1 | FileCheck --check-prefix=CHECK-NONEXISTENT %s
// RUN: not clang-query -c foo -f foo %s -- 2>&1 | FileCheck --check-prefix=CHECK-BOTH %s

// CHECK: unknown command: foo
// CHECK-NOT: unknown command: bar

// CHECK-NONEXISTENT: cannot open {{.*}}nonexistent.script
// CHECK-BOTH: cannot specify both -c and -f
