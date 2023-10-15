// RUN: %clang -### --entry test %s 2>&1 | FileCheck %s
// RUN: %clang -### --entry=test %s 2>&1 | FileCheck %s
// RUN: %clang -### -etest %s 2>&1 | FileCheck %s

// CHECK: "-e" "test"
