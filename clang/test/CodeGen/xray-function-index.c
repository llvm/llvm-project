// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -S -triple x86_64 -fxray-instrument -fxray-instruction-threshold=1 %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -triple x86_64 -fxray-instrument -fxray-instruction-threshold=1 -fno-xray-function-index %s -o - | FileCheck %s --check-prefix=NO

// CHECK: .section xray_fn_idx,"ao",@progbits,foo
// NO-NOT: .section xray_fn_idx

void foo(void) {}
