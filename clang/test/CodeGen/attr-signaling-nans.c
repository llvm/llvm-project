// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NO-SIGNALING
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsignaling-nans %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,SIGNALING
// Here msp430 is used as a target that do not support signaling NaNs.
// RUN: %clang_cc1 -triple msp430-unknown-unknown -fsignaling-nans -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NO-SIGNALING

float func_01(float x) {
  return x / 11.0F;
}

// expected-warning@*{{signaling NaNs are unsupported on this target}}

// CHECK: define{{.*}} float @func_01(float noundef{{.*}}) #[[ATTR:[0-9]+]] {


// SIGNALING: attributes #[[ATTR]] = {{{.*}} signaling_nans {{.*}}}
// NO-SIGNALING-NOT: attributes #[[ATTR]] = {{{.*}} signaling_nans {{.*}}}
