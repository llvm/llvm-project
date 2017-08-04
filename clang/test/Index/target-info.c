// RUN: c-index-test -test-print-target-info %s --target=i386-unknown-linux-gnu | FileCheck %s
// RUN: c-index-test -test-print-target-info %s --target=x86_64-unknown-linux-gnu | FileCheck --check-prefix=CHECK-1 %s
// CHECK: TargetTriple: i386-unknown-linux-gnu
// CHECK: PointerWidth: 32
// CHECK-1: TargetTriple: x86_64-unknown-linux-gnu
// CHECK-1: PointerWidth: 64
