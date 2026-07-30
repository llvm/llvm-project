// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -fxray-instrument -fxray-instruction-threshold=1 %s -o - \
// RUN:   | FileCheck %s

struct A {
  A();
  ~A();
};

A a;

// Check that the xray-instruction-threshold was applied
// CHECK: define internal void @_GLOBAL__sub_I_xray_global_init.cpp() [[NUX:#[0-9]+]] section ".text.startup" {
// CHECK: attributes [[NUX]] = { noinline nounwind {{.*}}"xray-instruction-threshold"="1"{{.*}} }
