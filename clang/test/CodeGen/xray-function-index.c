// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -S -triple x86_64 -fxray-instrument -fxray-instruction-threshold=1 -fxray-function-index %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -triple x86_64 -fxray-instrument -fxray-instruction-threshold=1 %s -o - | FileCheck %s --check-prefix=NO

// CHECK: .section xray_fn_idx,"awo",@progbits,foo
// NO-NOT: .section xray_fn_idx

void foo(void) {}
